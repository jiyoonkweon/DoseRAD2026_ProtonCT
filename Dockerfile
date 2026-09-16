# DoseRAD2026 Task 3 (proton, CT) submission container.
#
# The base image matches the organizers' example submission, which is already
# validated on the grading hardware.
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Unbuffered output: the last lines survive if the container is killed.
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt requirements-container.txt /opt/app/
# torch and numpy are already in the base image, so pip reports them as
# satisfied and skips them; the bounds in requirements.txt are chosen so that
# no CUDA-mismatched wheel is ever pulled in.
RUN python -m pip install --user --no-cache-dir --no-color \
    --requirement /opt/app/requirements-container.txt

COPY --chown=user:user protondose /opt/app/protondose
COPY --chown=user:user model /opt/app/model
ENV DOSERAD_MODEL_DIR=/opt/app/model

# Without this label the platform falls back to the legacy exec mode and the
# submission fails.
LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "-m", "protondose.server"]
