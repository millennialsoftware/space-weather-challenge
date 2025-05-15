# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists and install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python 3.10 and development tools
RUN apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3

RUN pip install evalai==1.3.13 lxml==4.6.2

RUN useradd -ms /bin/bash evalaiuser

# YOUR TOKEN GOES HERE
RUN mkdir -p /root/.evalai /home/evalaiuser/.evalai && \
    echo '{"token": "<EVALAI TOKEN>"}' > /root/.evalai/token.json && \
    cp /root/.evalai/token.json /home/evalaiuser/.evalai/token.json

RUN chmod -R 700 /root/.evalai /home/evalaiuser/.evalai && \
    chown -R evalaiuser:evalaiuser /home/evalaiuser/.evalai

WORKDIR /home/evalaiuser

ENV PATH="/home/evalaiuser/.local/bin:${PATH}"

ENTRYPOINT ["evalai"]
