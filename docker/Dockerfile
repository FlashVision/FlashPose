FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="FlashVision"
LABEL description="FlashPose: 2D/3D human pose estimation, hand/face keypoints, action & gesture recognition"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY flashpose/ flashpose/

RUN pip install --no-cache-dir -e ".[all]"

COPY configs/ configs/
COPY examples/ examples/

RUN flashpose check || true

ENTRYPOINT ["flashpose"]
CMD ["version"]
