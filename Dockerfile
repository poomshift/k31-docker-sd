ARG BASEIMAGE
ARG BASETAG

# STAGE FOR CACHING APT PACKAGE LIST
FROM ${BASEIMAGE}:${BASETAG} as stage_apt

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN \
    rm -rf /etc/apt/apt.conf.d/docker-clean \
	&& echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache \
	&& apt-get update

# STAGE FOR INSTALLING APT DEPENDENCIES
FROM ${BASEIMAGE}:${BASETAG} as stage_deps

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY aptDeps.txt /tmp/aptDeps.txt

# INSTALL APT DEPENDENCIES USING CACHE OF stage_apt
RUN \
    --mount=type=cache,target=/var/cache/apt,from=stage_apt,source=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt,from=stage_apt,source=/var/lib/apt \
    --mount=type=cache,target=/etc/apt/sources.list.d,from=stage_apt,source=/etc/apt/sources.list.d \
	apt-get install --no-install-recommends -y $(cat /tmp/aptDeps.txt) \
    && rm -rf /tmp/*

# ADD NON-ROOT USER user FOR RUNNING THE WEBUI
RUN \
    groupadd user \
    && useradd -ms /bin/bash user -g user \
    && echo "user ALL=NOPASSWD: ALL" >> /etc/sudoers


# STAGE FOR BUILDING APPLICATION CONTAINER
FROM stage_deps as stage_app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SD_WEBUI_VERSION

ENV \
    DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH \
    NVCC_FLAGS="--use_fast_math -DXFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD"\
    PATH=/usr/local/cuda-11.7/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"

# SWITCH TO THE GENERATED USER
WORKDIR /home/user
USER user

# CLONE AND PREPARE FOR THE SETUP OF SD-WEBUI
RUN \ 
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git -b ${SD_WEBUI_VERSION:-v1.6.0}


WORKDIR /home/user/stable-diffusion-webui
RUN python3 -m venv --system-site-packages /venv && \
    source /venv/bin/activate && \
    pip install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir xformers && \
    pip install httpx==0.24.0 && \
    pip install onnxruntime-gpu && \
    deactivate
WORKDIR /home/user/stable-diffusion-webui

Run git clone https://huggingface.co/embed/negative embeddings/negative && \
    git clone https://huggingface.co/embed/lora models/Lora/positive

# Clone the Automatic1111 Extensions
RUN git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet && \
    git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor && \
    git clone --depth=1 https://github.com/zanllp/sd-webui-infinite-image-browsing.git extensions/infinite-image-browsing && \
    git clone --depth=1 https://github.com/Bing-su/adetailer.git extensions/adetailer && \
    git clone --depth=1 https://github.com/Coyote-A/ultimate-upscale-for-automatic1111 extensions/ultimate-upscale-for-automatic1111 && \
    git clone --depth=1 https://github.com/richrobber2/canvas-zoom extensions/canvas-zoom && \
    git clone --depth=1 https://github.com/yankooliveira/sd-webui-photopea-embed extensions/sd-webui-photopea-embed && \
    git clone --depth=1 https://github.com/etherealxx/batchlinks-webui extensions/batchlinks-webui && \
    git clone --depth=1 https://github.com/continue-revolution/sd-webui-animatediff extensions/sd-webui-animatediff

RUN \
    mkdir /home/user/stable-diffusion-webui/outputs \
    && mkdir /home/user/stable-diffusion-webui/styles
    
RUN cd /home/user/stable-diffusion-webui/extensions/sd-webui-animatediff/model && \
    wget https://civitai.com/api/download/models/159987 --content-disposition && \
    cd /home/user/stable-diffusion-webui/models/Stable-diffusion && \
    wget https://civitai.com/api/download/models/148087 --content-disposition && \
    wget https://civitai.com/api/download/models/179525 --content-disposition && \
    cd /home/user/stable-diffusion-webui/models/Lora && \
    wget https://civitai.com/api/download/models/132876 --content-disposition && \
    mkdir -p /home/user/stable-diffusion-webui/models/ESRGAN && \
    cd /home/user/stable-diffusion-webui/models/ESRGAN && \
    wget https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth 
    
# Add inswapper model for the ReActor extension
RUN mkdir -p /home/user/stable-diffusion-webui/models/insightface && \
    cd /home/user/stable-diffusion-webui/models/insightface && \
    wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
    
WORKDIR /home/user/stable-diffusion-webui/extensions/sd-webui-controlnet/models
RUN wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors && \
    wget https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml && \
    wget https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors && \
    wget https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.yaml && \
    wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15.pth && \
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin && \
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin 
    
COPY ui-config.json /home/user/stable-diffusion-webui/

# RUN \
#     wget -O \
#         /home/user/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors \
#         https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors \
#     && ./webui.sh --xformers --skip-torch-cuda-test --no-download-sd-model --exit

RUN \
    ./webui.sh --xformers --skip-torch-cuda-test --no-download-sd-model --exit

# INCLUDE AUTO COMPLETION JAVASCRIPT
RUN \
    curl -o /home/user/stable-diffusion-webui/javascript/auto_completion.js \
        https://greasyfork.org/scripts/452929-webui-%ED%83%9C%EA%B7%B8-%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1/code/WebUI%20%ED%83%9C%EA%B7%B8%20%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1.user.js

# COPY entrypoint.sh
COPY --chmod=775 scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
USER root

# PORT AND ENTRYPOINT, USER SETTINGS
EXPOSE 7860
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# DOCKER IAMGE LABELING
LABEL title="Stable-Diffusion-Webui-Docker"
LABEL version=${SD_WEBUI_VERSION:-v1.6.0}

# ---------- BUILD COMMAND ----------
# DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain \
# SD_WEBUI_VERSION=v1.6.0 && \
#  docker build --no-cache \
# --build-arg BASEIMAGE=nvidia/cuda \
# --build-arg BASETAG=11.7.1-cudnn8-devel-ubuntu22.04 \
# --build-arg SD_WEBUI_VERSION=${SD_WEBUI_VERSION} \
# -t kestr3l/stable-diffusion-webui:${SD_WEBUI_VERSION} \
# -f Dockerfile .
