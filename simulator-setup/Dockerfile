FROM public.ecr.aws/docker/library/ubuntu@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

# Minimal build dependencies selected from upstream.Dockerfile and both simulators.
# All installations are inside this image, never on the host.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget build-essential cmake git bison flex zlib1g-dev \
    libgl-dev libglu1-mesa-dev libzstd-dev python3-dev xutils-dev gawk \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-8 cuda-cuobjdump-12-8 cuda-cudart-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*
# Both upstream runtime shims include cudaProfiler.h.
RUN apt-get update && apt-get install -y --no-install-recommends cuda-profiler-api-12-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
RUN nvcc --version && cuobjdump --version
WORKDIR /work
