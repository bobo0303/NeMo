FROM nvcr.io/nvidia/pytorch:23.10-py3   
# FROM nvcr.io/nvidia/nemo:24.05

ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libsndfile1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make && apt-get clean && rm -rf /var/lib/apt/lists  
  
RUN pip install --upgrade pip  
  
# 將 requirements.txt 複製到 Docker 映像中  
COPY requirements.txt .  
  
# 安裝 python packages  
RUN pip3 install -r requirements.txt  
  
# 設置工作目錄  
WORKDIR /app  
  
# 复制 app 资料夹到 Docker 映像中的 /app 目录  
COPY . /app  
  
# 设置环境变量  
ENV DOCKER_BUILDKIT=1 
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64  