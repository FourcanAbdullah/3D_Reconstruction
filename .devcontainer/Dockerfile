# Base: official COLMAP image (CUDA-enabled build)
FROM colmap/colmap:latest

# Enable universe; install MeshLab GUI + Python tooling + Qt/GL libs for WSLg
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository universe \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      meshlab \
      python3-full python3-venv python3-pip \
      # Qt/GL/X11/Wayland runtime pieces so GUI apps work under WSLg
      qtwayland5 libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 \
      libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 \
      libxfixes3 libsm6 libice6 libfontconfig1 libfreetype6 \
      xauth mesa-utils \
 && rm -rf /var/lib/apt/lists/*

# Create a venv for Python deps (avoids PEP 668 issues) and preinstall requirements
# NOTE: devcontainer.json sets build context to project root so COPY works.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# Make the venv the default Python in the container
ENV PATH="/opt/venv/bin:${PATH}"

# Work here by default; this will be bind-mounted by devcontainer.json
WORKDIR /workspace

