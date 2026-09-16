"""Grand Challenge request handling: input parsing and stacked output.

Output is one 4-D compressed MetaImage per slot. A single zlib stream is
structurally single-threaded, so frames are deflated independently in a worker
pool and the pieces concatenated behind one zlib header, with the Adler-32
checksums combined arithmetically. Standard decoders read the result unchanged;
selftest_parallel_deflate() checks that at startup.
"""
from __future__ import annotations

import concurrent.futures as _cf
import gc
import glob
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch

from .geometry import patient_phys_coords
from .inference import (CUTOFF_GUARD_RATIO, CUTOFF_SKIPPED, MAX_INFLIGHT,
                        NORM_SCALE, N_COMPRESS_WORKERS, Beamlet, apply_cutoff,
                        make_density, predict_beamlets)

INPUT_PATH = Path(os.environ.get("DOSERAD_IN", "/input"))
OUTPUT_PATH = Path(os.environ.get("DOSERAD_OUT", "/output"))
IMAGE_DIR_SLUG = "ct"
METADATA_NAME = "stacked-proton-beam-level-metadata"
OUTPUT_SLUG_BASE = "stacked-radiation-dose-map"
NUM_IO_SLOTS = 10

try:
    from isal import isal_zlib as _Z              # ISA-L: zlib-compatible, faster
    _Z_NAME = "isal"
except ImportError:
    import zlib as _Z
    _Z_NAME = "zlib"
import zlib as _ZREF                              # self-test checks against stdlib

# Level 1 rather than 0: the ISA-L level 0 path skips Huffman optimization, which
# on this data is both slower and about 40% larger.
_Z_LEVEL = 1
_DEFLATED = getattr(_Z, "DEFLATED", 8)
_FULL_FLUSH = getattr(_Z, "Z_FULL_FLUSH", 3)
_FINISH = getattr(_Z, "Z_FINISH", 4)

_N_FIN = 4                                        # slot finalizers (I/O bound)

_ADLER_BASE = 65521


def _adler32_combine(a1: int, a2: int, len2: int) -> int:
    """Port of zlib's adler32_combine_, to merge per-frame checksums in order."""
    rem = len2 % _ADLER_BASE
    s1 = a1 & 0xffff
    s2 = (rem * s1) % _ADLER_BASE
    s1 += (a2 & 0xffff) + _ADLER_BASE - 1
    s2 += ((a1 >> 16) & 0xffff) + ((a2 >> 16) & 0xffff) + _ADLER_BASE - rem
    if s1 >= _ADLER_BASE:
        s1 -= _ADLER_BASE
    if s1 >= _ADLER_BASE:
        s1 -= _ADLER_BASE
    if s2 >= (_ADLER_BASE << 1):
        s2 -= _ADLER_BASE << 1
    if s2 >= _ADLER_BASE:
        s2 -= _ADLER_BASE
    return s1 | (s2 << 16)


def _deflate_final_block() -> bytes:
    """An empty deflate block with the final bit set: terminates the stream."""
    co = _Z.compressobj(_Z_LEVEL, _DEFLATED, -15)
    return co.compress(b"") + co.flush(_FINISH)


def _compress_frame(sub: np.ndarray, bbox, full_shape) -> tuple:
    """Worker: expand a sub-volume to the full frame and deflate it.

    Ending on Z_FULL_FLUSH closes the frame on a byte boundary, leaves the final
    bit clear and resets the dictionary, so concatenating frames in order yields
    one valid deflate stream.  zlib releases the GIL, so these threads are truly
    parallel.  Returns (compressed, adler32, uncompressed_bytes, worker_ms).
    """
    t0 = time.perf_counter()
    nz, ny, nx = full_shape
    x0, x1, y0, y1, z0, z1 = bbox
    full = np.zeros((nz, ny, nx), np.float32)     # calloc: outside the box is 0
    full[z0:z1, y0:y1, x0:x1] = sub
    mv = memoryview(full).cast("B")               # avoid a tobytes() copy
    co = _Z.compressobj(_Z_LEVEL, _DEFLATED, -15)
    comp = co.compress(mv) + co.flush(_FULL_FLUSH)
    return comp, _Z.adler32(mv), mv.nbytes, (time.perf_counter() - t0) * 1e3


def selftest_parallel_deflate() -> None:
    """Verify the concatenated stream round-trips through stdlib zlib."""
    rng = np.random.default_rng(0)
    frames = [np.zeros((9, 7, 5), np.float32),
              rng.standard_normal((9, 7, 5)).astype(np.float32),
              np.zeros((9, 7, 5), np.float32)]
    parts, adler = [], 1
    for i, fr in enumerate(frames):
        c, a, n, _ = _compress_frame(np.ascontiguousarray(fr), (0, 5, 0, 7, 0, 9), (9, 7, 5))
        parts.append(c)
        adler = a if i == 0 else _adler32_combine(adler, a, n)
    stream = b"\x78\x01" + b"".join(parts) + _deflate_final_block() \
        + adler.to_bytes(4, "big")
    ref = b"".join(bytes(memoryview(np.ascontiguousarray(f)).cast("B")) for f in frames)
    if _ZREF.decompress(stream) != ref:
        raise RuntimeError("parallel deflate self-test failed: stream or adler mismatch")
    print(f"[zlib] backend={_Z_NAME} level={_Z_LEVEL} workers={N_COMPRESS_WORKERS} "
          f"(self-test passed)", flush=True)


def parse_metadata(meta: List[Dict]) -> List[Dict]:
    """Flatten the beams/rays/beamlets tree into one record per beamlet."""
    records: List[Dict] = []
    for image in meta:
        image_idx = int(image["image_file_idx"])
        for beam in image.get("beams", []):
            for ray in beam.get("rays", []):
                src, tgt = ray["ray_source"], ray["ray_target"]
                for bl in ray.get("beamlets", []):
                    oi = bl["output_info"]
                    records.append({
                        "output_file_idx": int(oi["output_file_idx"]),
                        "idx_in_output": int(oi["idx_in_output"]),
                        "minimum_cutoff": oi.get("minimum_cutoff"),
                        "image_file_idx": image_idx,
                        "item": Beamlet(src, tgt, bl["energy"]),
                    })
    if not records:
        raise ValueError("no beamlets found in the metadata")
    return records


def load_input_image(image_file_idx: int) -> sitk.Image:
    n = image_file_idx + 1
    loc = INPUT_PATH / "images" / f"radiation-dose-calculation-source-{IMAGE_DIR_SLUG}-image-{n}"
    files = sorted(glob.glob(str(loc / "*.mha")))
    if not files:
        raise FileNotFoundError(f"no input image for index {n} under {loc}")
    return sitk.ReadImage(files[0])


_COMP_POOL: _cf.ThreadPoolExecutor = None
_FIN_POOL: _cf.ThreadPoolExecutor = None
_FIN_FUTS: List[_cf.Future] = []


def _slot_dir(slot_index0: int) -> Path:
    d = OUTPUT_PATH / "images" / f"{OUTPUT_SLUG_BASE}-{slot_index0 + 1}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_placeholder(slot_index0: int) -> None:
    """Unused slots still have to exist; a 1x1 image is the smallest valid file."""
    sitk.WriteImage(sitk.Image(1, 1, sitk.sitkFloat32),
                    str(_slot_dir(slot_index0) / "output.mha"), useCompression=True)


def _finalize_slot(slot_index0: int, futs: List[_cf.Future], n: int,
                   meta: Dict, prof: Dict) -> None:
    """Concatenate the compressed frames, combine checksums, write the file.

    The header needs CompressedDataSize up front, but that is the sum of the
    piece lengths, so no scratch round-trip is needed.
    """
    t0 = time.perf_counter()
    adler, total, comp_ms = 1, 0, 0.0
    parts = []
    csize = 2                                        # zlib header
    for i, fut in enumerate(futs):
        comp, ad, nbytes, tms = fut.result()         # worker exceptions surface here
        parts.append(comp)
        csize += len(comp)
        adler = ad if i == 0 else _adler32_combine(adler, ad, nbytes)
        total += nbytes
        comp_ms += tms
    tail = _deflate_final_block()
    csize += len(tail) + 4                           # final block + adler

    D4 = np.eye(4)
    D4[:3, :3] = meta["D"]
    sx, sy, sz = meta["sp"]
    ox, oy, oz = meta["org"]
    nx, ny, nz = meta["size"]
    hdr = (f"ObjectType = Image\nNDims = 4\nBinaryData = True\n"
           f"BinaryDataByteOrderMSB = False\nCompressedData = True\n"
           f"CompressedDataSize = {csize}\n"
           f"TransformMatrix = {' '.join(f'{v:g}' for v in D4.flatten())}\n"
           f"Offset = {ox:g} {oy:g} {oz:g} 0\n"
           f"CenterOfRotation = 0 0 0 0\n"
           f"ElementSpacing = {sx:g} {sy:g} {sz:g} 1\n"
           f"DimSize = {nx} {ny} {nz} {n}\n"
           f"AnatomicalOrientation = ????\n"
           f"ElementType = MET_FLOAT\nElementDataFile = LOCAL\n")
    with open(meta["path"], "wb") as out:
        out.write(hdr.encode("ascii"))
        out.write(b"\x78\x01")
        for c in parts:
            out.write(c)
        out.write(tail)
        out.write(adler.to_bytes(4, "big"))

    print(f"[prof] slot {slot_index0 + 1}: produce {prof.get('produce_s', 0):.2f}s "
          f"(stall {prof.get('stall_s', 0):.2f}s) | compress {comp_ms / 1e3:.2f}s "
          f"({N_COMPRESS_WORKERS}w {_Z_NAME} L{_Z_LEVEL}) "
          f"| finalize {time.perf_counter() - t0:.2f}s "
          f"| csize {csize / 2 ** 20:.1f}MiB", flush=True)


def write_stack_parallel(slot_index0: int, frames, n: int,
                         reference: sitk.Image, prof: Dict) -> None:
    """Consume (sub_volume, bbox) frames in output order and queue them.

    The main thread only submits, so it stays free to drive the GPU. The
    in-flight window bounds queued sub-volume memory; time spent waiting on it
    means compression is not keeping up.
    """
    d = _slot_dir(slot_index0)
    nx, ny, nz = reference.GetSize()
    meta = dict(path=str(d / "output.mha"),
                D=np.asarray(reference.GetDirection(), float).reshape(3, 3),
                sp=tuple(reference.GetSpacing()), org=tuple(reference.GetOrigin()),
                size=(nx, ny, nz))
    futs: List[_cf.Future] = []
    inflight: deque = deque()
    t_stall = 0.0
    t0 = time.perf_counter()
    for sub, bbox in frames:
        x0, x1, y0, y1, z0, z1 = bbox
        if not (0 <= x0 <= x1 <= nx and 0 <= y0 <= y1 <= ny and 0 <= z0 <= z1 <= nz):
            raise RuntimeError(f"bounding box outside the patient grid: {bbox}")
        if len(inflight) >= MAX_INFLIGHT:
            ts = time.perf_counter()
            inflight.popleft().result()
            t_stall += time.perf_counter() - ts
        fut = _COMP_POOL.submit(_compress_frame, sub, bbox, (nz, ny, nx))
        futs.append(fut)
        inflight.append(fut)
    prof["produce_s"] = time.perf_counter() - t0
    prof["stall_s"] = t_stall
    if len(futs) != n:
        raise RuntimeError(f"frame count mismatch: {len(futs)} vs {n}")
    _FIN_FUTS.append(_FIN_POOL.submit(_finalize_slot, slot_index0, futs, n, meta, prof))


def run(model: Dict) -> int:
    global _COMP_POOL, _FIN_POOL

    _COMP_POOL = _cf.ThreadPoolExecutor(max_workers=N_COMPRESS_WORKERS, thread_name_prefix="comp")
    _FIN_POOL = _cf.ThreadPoolExecutor(max_workers=_N_FIN, thread_name_prefix="fin")
    _FIN_FUTS.clear()

    device = model["device"]
    with open(INPUT_PATH / f"{METADATA_NAME}.json") as f:
        records = parse_metadata(json.load(f))
    print(f"[run] {len(records)} beamlet(s) requested", flush=True)

    slots: List[List[Dict]] = [[] for _ in range(NUM_IO_SLOTS)]
    for r in records:
        slots[r["output_file_idx"]].append(r)
    used = [(i, rs) for i, rs in enumerate(slots) if rs]
    used.sort(key=lambda t: (t[1][0]["image_file_idx"], t[0]))

    cur_image_idx = None
    image = density_gpu = phys = None
    cutoff_seen, dose_max_seen = set(), 0.0

    try:
        for slot_index0, recs in used:
            image_idx = recs[0]["image_file_idx"]
            if any(r["image_file_idx"] != image_idx for r in recs):
                raise ValueError(f"output slot {slot_index0 + 1} mixes image indices")
            if image_idx != cur_image_idx:
                del density_gpu, phys
                density_gpu = phys = None
                torch.cuda.empty_cache()
                image = load_input_image(image_idx)
                density_gpu = make_density(image, model["hlut"], device)
                phys = patient_phys_coords(image, device=device)
                cur_image_idx = image_idx
                print(f"[run] image {image_idx + 1}: size={image.GetSize()} "
                      f"spacing={image.GetSpacing()}", flush=True)

            recs.sort(key=lambda r: r["idx_in_output"])
            if [r["idx_in_output"] for r in recs] != list(range(len(recs))):
                raise ValueError(f"slot {slot_index0 + 1}: idx_in_output is not 0..N-1")

            t0 = time.time()
            stats = {"max": 0.0}

            def _frames(recs=recs):
                gen = predict_beamlets(model, [r["item"] for r in recs], image,
                                       density_gpu, phys)
                for r, (sub, bbox) in zip(recs, gen):
                    sub *= NORM_SCALE                     # to absolute Gy, in place
                    stats["max"] = max(stats["max"], float(sub.max()) if sub.size else 0.0)
                    if r["minimum_cutoff"] is not None:
                        cutoff_seen.add(float(r["minimum_cutoff"]))
                    yield apply_cutoff(sub, r["minimum_cutoff"]), bbox

            n = len(recs)
            prof: Dict = {}
            write_stack_parallel(slot_index0, _frames(), n, image, prof)
            dose_max_seen = max(dose_max_seen, stats["max"])
            gc.collect()                    # release the finished slot's frames
            dt = time.time() - t0
            print(f"[run] slot {slot_index0 + 1}: {n} beamlet(s) produced in {dt:.2f}s "
                  f"({dt / max(n, 1) * 1000:.0f} ms/beamlet; compression in background)",
                  flush=True)

        for f in _FIN_FUTS:                    # re-raise any background failure
            f.result()
    finally:
        _COMP_POOL.shutdown(wait=True)
        _FIN_POOL.shutdown(wait=True)

    for slot_index0 in range(NUM_IO_SLOTS):
        if not slots[slot_index0]:
            write_placeholder(slot_index0)

    print(f"[diag] minimum_cutoff values seen: {sorted(cutoff_seen)}", flush=True)
    if CUTOFF_SKIPPED["n"]:
        print(f"[diag] cutoff skipped for {CUTOFF_SKIPPED['n']} beamlet(s): "
              f"{sorted(CUTOFF_SKIPPED['cutoffs'])} exceeded "
              f"{CUTOFF_GUARD_RATIO} x predicted max {CUTOFF_SKIPPED['max_seen']:.6g}",
              flush=True)
    print(f"[diag] max predicted dose (Gy): {dose_max_seen:.6g}", flush=True)
    if cutoff_seen and dose_max_seen > 0 and min(cutoff_seen) >= dose_max_seen:
        print("[diag] cutoff is at or above the predicted maximum: output is all zero",
              flush=True)
    return 0

