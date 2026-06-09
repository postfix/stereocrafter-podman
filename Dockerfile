FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# OS packages + python 3.8 from deadsnakes (StereoCrafter pins torch==2.0.1 which needs py<=3.11; 3.8 is the README spec)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
        git git-lfs ffmpeg \
        build-essential ninja-build pkg-config \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-venv python3.8-distutils \
 && curl -fsSL https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 \
 && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.8 /usr/local/bin/python3 \
 && git lfs install --system \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone StereoCrafter (pin to known-good commit) + submodules
ARG STEREOCRAFTER_REF=e30707728bd2f52cec2de8630098a7ae4f95d27c
RUN git clone https://github.com/TencentARC/StereoCrafter.git /opt/StereoCrafter \
 && cd /opt/StereoCrafter \
 && git checkout ${STEREOCRAFTER_REF} \
 && git submodule update --init --recursive

# Torch + xformers must be installed against CUDA 11.8 wheels before the rest of requirements
RUN python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel \
 && python -m pip install \
        torch==2.0.1 torchvision==0.15.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
 && python -m pip install xformers==0.0.20 \
 && python -m pip install -r /opt/StereoCrafter/requirements.txt \
 && python -m pip install "huggingface_hub[cli,hf_transfer]"

# Build Forward-Warp CUDA extension (sm_86 for RTX 3090 set via TORCH_CUDA_ARCH_LIST)
# Use pip install so the egg lands in dist-packages (deadsnakes python's site-packages isn't on sys.path)
WORKDIR /opt/StereoCrafter/dependency/Forward-Warp
RUN cd Forward_Warp/cuda && pip install --no-build-isolation -v . \
 && cd ../.. && pip install --no-build-isolation -v .

# Sanity check that the extension imports
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
import xformers; print('xformers', xformers.__version__); \
from Forward_Warp import forward_warp; print('Forward_Warp ok')"

WORKDIR /opt/StereoCrafter
ENV PYTHONPATH=/opt/StereoCrafter

# The base nvidia/cuda image prints a license banner on every run; that pollutes
# stdout-captured tool output (ffprobe, etc.). Override the entrypoint so commands
# print only their own output.
ENTRYPOINT []
CMD ["bash"]
