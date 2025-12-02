FROM colmap/colmap:latest
SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# --- System deps (incl. toolchain for building wheels like pyliblzfse) ---
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg git curl ca-certificates \
#     build-essential python3-dev pkg-config \
#     liblzfse-dev \
#     libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
#     libxcb-keysyms1 libxcb-render-util0 libgl1 libglu1-mesa mesa-utils xauth \
#     qtwayland5 libegl1 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 \
#     libxfixes3 libsm6 libice6 libfontconfig1 libfreetype6 \
#     libxcb1 libx11-xcb1 libglib2.0-0 meshlab && \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core tools
    ffmpeg git curl ca-certificates \
    build-essential python3-dev pkg-config \
    liblzfse-dev \
    mesa-utils xvfb xauth \
\
    # X11 / Wayland / Qt / GUI deps
    qt5-qmake qtbase5-dev qtchooser qtwayland5 \
    libegl1 libgl1 libglu1-mesa \
\
    # XCB stack (for Qt, COLMAP, Nerfstudio GUI tools)
    libxcb1 libx11-xcb1 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-xinerama0 libxkbcommon-x11-0 \
\
    # Standard X11 libs required by COLMAP, Open3D, etc.
    libx11-6 libxext6 libxrender1 libxi6 \
    libxrandr2 libxinerama1 libxfixes3 \
\
    # Font + general deps
    libsm6 libice6 libfontconfig1 libfreetype6 \
\
    # Utility GUI applications (optional but useful)
    meshlab \
\
    && rm -rf /var/lib/apt/lists/*



# --- Micromamba (lightweight conda) ---
ENV MAMBA_ROOT_PREFIX=/opt/conda
# Install necessary utilities (curl, tar, bzip2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 tar \
    && rm -rf /var/lib/apt/lists/*

# Install Micromamba from the latest stable release
RUN curl -L https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
    -o /usr/local/bin/micromamba && \
    chmod +x /usr/local/bin/micromamba


# --- Create env "ns" with Python 3.10 ---
RUN micromamba create -y -n ns python=3.10 && \
    micromamba clean -a -y
# Make env default on login/shell
SHELL ["/usr/local/bin/micromamba", "run", "-n", "ns", "/bin/bash", "-lc"]

# --- Install PyTorch (CUDA build) + Nerfstudio *inside* the env ---
# Change cu121->cu118 if your host only supports CUDA 11.8
ARG TORCH_CUDA=cu121
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
ARG TORCHAUDIO_VERSION=2.3.1
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
RUN apt-get update && apt-get install -y build-essential cmake

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir --extra-index-url "${TORCH_INDEX_URL}" \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} && \
    python -m pip install --no-cache-dir "nerfstudio==1.*" rich click
# Optional: Gaussian splatting plugin
# RUN python -m pip install --no-cache-dir gsplat

# --- Workspace ---
WORKDIR /workspace
ENV NERFSTUDIO_ENABLE_VIEWER=0
# Make sure interactive terminals also drop into the env
ENV PATH="$MAMBA_ROOT_PREFIX/envs/ns/bin:$PATH"
# Ensure Qt uses X11 for GUI

CMD ["/bin/bash"]
