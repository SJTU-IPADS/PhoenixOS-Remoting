# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
# Automatically set by buildx
ARG TARGETPLATFORM
# translating Docker's TARGETPLATFORM into miniconda arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
COPY requirements.txt .
RUN chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya

FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule update --init --recursive --jobs 0

FROM conda as build
WORKDIR /opt/pytorch
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/pytorch /opt/pytorch
RUN --mount=type=cache,target=/opt/ccache \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all -cudart shared" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install

FROM conda as conda-installs
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.3
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch-nightly
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
# Automatically set by buildx
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -y python=${PYTHON_VERSION}
ARG TARGETPLATFORM
# On arm64 we can only install wheel packages
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  pip install --extra-index-url https://download.pytorch.org/whl/cpu/ torch torchvision torchtext ;; \
         *)              /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch torchvision torchtext "cudatoolkit=${CUDA_VERSION}"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic

FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=build /opt/conda /opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
        rpcbind \
        git \
        automake \
        libtool \
        libssl-dev \
        inetutils-ping \
        vim \
        libgl1-mesa-dev \
        gdb \
        wget \
        rpm2cpio \
        cpio \
        cuda-samples-11-3 \
        openssl1.1 \
        libncurses5-dev \
        libexpat1-dev \
        liblzma-dev \
        flex \
        bison \
        texinfo \
        libelf-dev \
        libvdpau-dev \
        libgles2-mesa-dev \
        binutils-dev && \
    rm -rf /var/lib/apt/lists/*

# Set MOFED version, OS version and platform
ENV MOFED_VERSION 5.1-2.3.7.1
ENV OS_VERSION ubuntu18.04
ENV PLATFORM x86_64
ENV OFED https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.1-2.3.7.1/MLNX_OFED_LINUX-5.1-2.3.7.1-ubuntu18.04-x86_64.tgz
ENV OFED_FILE MLNX_OFED_LINUX-5.1-2.3.7.1-ubuntu18.04-x86_64.tgz
ENV OFED_PATH MLNX_OFED_LINUX-5.1-2.3.7.1-ubuntu18.04-x86_64

RUN apt-get update
RUN apt-get -y install apt-utils
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential cmake tcsh tcl tk \
	        make git curl vim wget ca-certificates \
		        iputils-ping net-tools ethtool \
			        perl lsb-release python-libxml2 \
				        iproute2 pciutils libnl-route-3-200 \
					        kmod libnuma1 lsof openssh-server \
						        swig libelf1 automake libglib2.0-0 \
							        autoconf graphviz chrpath flex libnl-3-200 m4 \
								        debhelper autotools-dev gfortran libltdl-dev && \
                                            dpatch udev pkg-config libnl-3-dev libnl-route-3-dev && \
									            rm -rf /rm -rf /var/lib/apt/lists/*
RUN mv /opt/conda/bin/python /opt/conda/bin/python.bak
RUN wget --quiet ${OFED} && \
    tar -xvf ${OFED_FILE} && \
        ${OFED_PATH}/mlnxofedinstall --user-space-only --without-fw-update -q && \
		    rm -rf ${OFED_FILE}
RUN mv /opt/conda/bin/python.bak /opt/conda/bin/python

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
        mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

