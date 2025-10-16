
FROM colmap/colmap:latest


RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository universe \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      meshlab \
      python3-full python3-venv python3-pip \

      qtwayland5 libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 \
      libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 \
      libxfixes3 libsm6 libice6 libfontconfig1 libfreetype6 \
      xauth mesa-utils \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt


ENV PATH="/opt/venv/bin:${PATH}"


WORKDIR /workspace

