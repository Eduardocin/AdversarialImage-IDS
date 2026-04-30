FROM bvlc/caffe:cpu

LABEL org.opencontainers.image.title="AdversarialImage-IDS"
LABEL org.opencontainers.image.description="Legacy Caffe/TensorFlow environment for DeepDetector replication"

ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_ROOT=/workspace
ENV PYTHONUNBUFFERED=1

WORKDIR ${PROJECT_ROOT}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python-tk \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-legacy.txt /tmp/requirements-legacy.txt

RUN pip install --no-cache-dir --upgrade "pip<21" "setuptools<45" wheel \
    && pip install --no-cache-dir --ignore-installed -r /tmp/requirements-legacy.txt

ENV PYTHONPATH=/opt/caffe/python:${PROJECT_ROOT}/src/original/DeepDetector:${PYTHONPATH}

CMD ["/bin/bash"]
