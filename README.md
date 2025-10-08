What it is

How to run with VS Code Dev Containers

How to build/run with Docker
Open a terminal on your Ubuntu desktop and run:
    sudo apt update
    sudo apt install -y docker.io docker-buildx docker-compose-plugin
    sudo usermod -aG docker $USER
    newgrp docker

    sudo apt install -y nvidia-driver-535 nvidia-container-toolkit
    sudo nvidia-ctk runtime configure
    sudo systemctl restart docker
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
Install VS Code + Dev Container extensions
    git clone
    code
in docker:
    echo $DISPLAY          # should print :0 or :1
    xclock                 # optional GUI test (should pop up a clock)
    python app.py 
GPU/CPU behavior

WSLg vs X11 notes

SSH:
To display anything run:
sudo apt update
sudo apt install -y libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util

unset WAYLAND_DISPLAY
export QT_QPA_PLATFORM=xcb
unset QT_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt6/plugins/platforms
export LIBGL_ALWAYS_INDIRECT=1
