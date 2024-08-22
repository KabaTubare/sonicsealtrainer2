# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    cmake \
    ninja-build \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    && apt-get clean

# Set environment variable USER to root
ENV USER=root

# Create the necessary directories before copying
RUN mkdir -p /app/config/dset /app/config/solver /app/config/model

# Copy the configuration directories
COPY ./config/dset /app/config/dset
COPY ./config/solver /app/config/solver
COPY ./config/model /app/config/model

# Copy the entire audiocraft directory and other necessary files into the container
COPY ./audiocraft /app/audiocraft
COPY ./scripts /app/scripts
COPY ./requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install torch torchaudio torchvision torchtext \
    && pip install -r requirements.txt

# Install the 'av' Python package for working with audio-visual formats
RUN pip install av

# Install audiocraft as a Python package (editable install)
RUN pip install -e /app/audiocraft

# Verify that the configuration directories and files are copied correctly
RUN ls -l /app/config/dset /app/config/solver /app/config/model

# Verify that all dependencies are installed
RUN pip list

# Set the command to run the training script
CMD ["python3", "scripts/train.py"]
