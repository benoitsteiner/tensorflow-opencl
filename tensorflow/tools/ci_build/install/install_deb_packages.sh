#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
ubuntu_version=$(cat /etc/issue | grep -i ubuntu | awk '{print $2}' | \
  awk -F'.' '{print $1}')

# Install dependencies from ubuntu deb repository.
apt-get update

set +e
ffmpeg_location=$(which ffmpeg)
if [[ -z "$ffmpeg_location"  && "$ubuntu_version" == "14" ]]; then
  set -e
  # specifically for trusty linked from ffmpeg.org
  add-apt-repository -y ppa:mc3man/trusty-media
  apt-get update
  apt-get dist-upgrade -y
  apt-get install -y ffmpeg libav-tools
else
  set -e
  apt-get install -y ffmpeg libav-tools
fi

apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    libtool \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    pkg-config \
    python-dev \
    python-pip \
    python-virtualenv \
    python3-dev \
    python3-pip \
    rsync \
    sudo \
    swig \
    unzip \
    wget \
    zip \
    zlib1g-dev

# Install ca-certificates, and update the certificate store.
apt-get install -y ca-certificates-java
update-ca-certificates -f

apt-get clean
rm -rf /var/lib/apt/lists/*
