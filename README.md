
# How to Run on Ubuntu

## 1. Requirements
- Ubuntu system with an NVIDIA GPU and CUDA 12.9  
- Docker version **17.12.0 or later**  
- VS Code with the **Dev Containers** extension installed  
- User must be added to the `docker` group  
  - (Log out and back in after adding the user)  
- NVIDIA drivers installed and `nvidia-smi` working on host  and in container
- NVIDIA Docker Toolkit installed  
- CUDA Toolkit installed on the host  

---

## 2. Clone Repository and Open in VS Code
1. Clone the repository.  
2. Open the project folder in VS Code.  
3. Ensure Docker is running.  
4. VS Code will detect the `.devcontainer` folder automatically.

---

## 3. Configure GPU Access

Edit:

```

.devcontainer/devcontainer.json

````

### For GPU-enabled machines
If not already there add the following as the first entry under `"runArgs"`:

```json
"runArgs": [
    "--gpus=all"
]
````

---

## 4. Tutorials to set up GPU in Docker(GPU + Docker)

* [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* [https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/](https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/)
* [https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/](https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/)
* [https://www.devzero.io/blog/docker-gpu](https://www.devzero.io/blog/docker-gpu)

---
## 5. Setting Up NVIDIA Docker (Optional Reference)

Run these **on the host system (outside the container)**:


```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
 | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo apt-get install -y nvidia-cuda-toolkit
sudo systemctl restart docker
```

Verify GPU:

```bash
nvidia-smi
```


---

## 6. Build and Open the Dev Container

In VS Code:

```
Ctrl + Shift + P → "Dev Containers: Rebuild and Reopen in Container"
```

---

## 7. Enable GUI Applications from Docker

Run the following on the **host**:

```bash
sudo apt install x11-xserver-utils
xhost +local:docker
```

---
## 8. Run

Open new terminal in the container once its finished building and run:

```bash
python -m app
```
---

## 9. To view Point Clouds, compare PointClouds and check stats

Install CloudCompare
[https://snapcraft.io/install/cloudcompare/ubuntu#install](https://snapcraft.io/install/cloudcompare/ubuntu#install)
```bash
sudo snap install cloudcompare
```
Run
```bash
cloudcompare.CloudCompare
```
---

## 9. To find Pipeline files and  Find PLY Point Cloud Results:
Pipelines Can be found in /workspace/app/pipelines

To change any parameters in the pipelines simply modify the cli command in the python file with a existing parameter.

Point Cloud results can be found in the outputs directory(output will be created on the first run of any of the methods)

Example Path to COLMAP Point cloud: /workspace/outputs/run_20251211-050837/method_a/colmap_workspace/dense/fused.ply

Example Path to NerfStudio Point Cloud: /workspace/outputs/run_20251210-022551/method_b/exports/pointcloud/point_cloud.ply

Run CloudCompare and input the PLY files to view point cloud. 

# CUDA Out Of Memory Error:
Add these lines inside ns-train (around line 106)
```python
        "--pipeline.datamanager.train-num-rays-per-batch", "2048",
        "--pipeline.datamanager.eval-num-rays-per-batch", "1024"
```
# SSH Display Fix (Qt / XCB Errors)

If using SSH or running GUI apps remotely, run these **inside the container**:

```bash
sudo apt update
sudo apt install -y libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 \
libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util
```

Then set the required environment variables:

```bash
unset WAYLAND_DISPLAY
export QT_QPA_PLATFORM=xcb
unset QT_PLUGIN_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt6/plugins/platforms
export LIBGL_ALWAYS_INDIRECT=1
```

---
