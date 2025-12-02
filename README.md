What it is

How to run with VS Code Dev Containers
How to Run in Ubuntu:
After Cloning the Repo, Open the Folder in VSCode then make sure to install the Dev Containers Extension in VScode. Also Install Docker Version 17.12.0 or later and make sure the user is in the docker profile.(you may need to log out and log in for changes to take place)
Make sure that you have nvidia docker, nvidia smi, nvidia-cuda-toolkit installed in the main system.
# Add NVIDIA's package repositories
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf > /dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF

# Download and install the GPG key for NVIDIA's repository
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

# Add the NVIDIA package repositories (replace 'ubuntu18.04' with your version, e.g., 'ubuntu20.04')
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update and install NVIDIA Docker toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

nvidia-smi

sudo apt-get install -y nvidia-cuda-toolkit

Then in VScode do ctrl + alt + P and run Build Container

How to build/run with Docker
sudo apt install x11-xserver-utils
Make sure to run "xhost +local:docker" in host machine to see colmap gui
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
