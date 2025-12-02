
#How to Run in Ubuntu:
##Required to have GPU to work!
##After Cloning the Repo, Open the Folder in VSCode then make sure to install the Dev Containers Extension in VScode. Also Install Docker Version 17.12.0 or later and make sure the user is in the docker profile.(you may need to log out and log in for changes to take place)
##Make sure that you have nvidia docker, nvidia smi, nvidia-cuda-toolkit installed in the main system.
##Before opening the devcontainer on a machine with a GPU, go to the .devcontainer folder
open devcontainer.json
inside "runArgs":[]
add to the first line: "--gpus=all", 
--On a CPU-only / old-GPU machine:
--remove:  "--gpus=all",
##How I added added GPU to container (if you want to use this make sure you are outside of container, in root folder.) there are tuturials to set this up below.
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf > /dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

nvidia-smi

sudo apt-get install -y nvidia-cuda-toolkit

Then in VScode do ctrl + shift + P and run Build Container
###Outside the Container, in the main computer run the following to see GUI:

sudo apt install x11-xserver-utils
xhost +local:docker

##Some tutorials on installing and making visible Nvidia GPU in docker for your reference:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/
https://www.devzero.io/blog/docker-gpu

SSH:
To display anything run:
sudo apt update
sudo apt install -y libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util

unset WAYLAND_DISPLAY
export QT_QPA_PLATFORM=xcb
unset QT_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt6/plugins/platforms
export LIBGL_ALWAYS_INDIRECT=1
