What it is

How to run with VS Code Dev Containers
How to Run in Ubuntu:
After Cloning the Repo, Open the Folder in VSCode then make sure to install the Dev Containers Extension in VScode. Also Install Docker Version 17.12.0 or later and make sure the user is in the docker profile.(you may need to log out and log in for changes to take place)
Make sure that you have nvidia docker, nvidia smi, nvidia-cuda-toolkit installed.

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
