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

# Install pip packages from whl files to avoid the time-consuming process of
# building from source.

pip install wheel
pip3 install wheel

# Install six.
pip install --upgrade six==1.10.0
pip3 install --upgrade six==1.10.0

# Install protobuf.
pip install --upgrade protobuf==3.0.0
pip3 install --upgrade protobuf==3.0.0

# Remove obsolete version of six, which can sometimes confuse virtualenv.
rm -rf /usr/lib/python3/dist-packages/six*

set +e
# Use pip to install numpy to a modern version, instead of 1.8.2 that comes
# with apt-get in ubuntu:14.04.
NUMPY_VERSION="1.11.0"
numpy_ver_flat=$(echo $NUMPY_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_numpy_ver=$(python -c "import numpy; print(numpy.__version__)")
local_numpy_ver_flat=$(echo $local_numpy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_numpy_ver_flat ]]; then
  local_numpy_ver_flat=0
fi
if (( $local_numpy_ver_flat < $numpy_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/06/92/3c786303889e6246971ad4c48ac2b4e37a1b1c67c0dc2106dc85cb15c18e/numpy-1.11.0-cp27-cp27mu-manylinux1_x86_64.whl#md5=6ffb66ff78c28c55bfa09a2ceee487df
  mv numpy-1.11.0-cp27-cp27mu-manylinux1_x86_64.whl \
     numpy-1.11.0-cp27-none-linux_x86_64.whl
  pip install numpy-1.11.0-cp27-none-linux_x86_64.whl
  rm numpy-1.11.0-cp27-none-linux_x86_64.whl
fi

set +e
local_numpy_ver=$(python3 -c "import numpy; print(numpy.__version__)")
local_numpy_ver_flat=$(echo $local_numpy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_numpy_ver_flat ]]; then
  local_numpy_ver_flat=0
fi
if (( $local_numpy_ver_flat < $numpy_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/ea/ca/5e48a68be496e6f79c3c8d90f7c03ea09bbb154ea4511f5b3d6c825cefe5/numpy-1.11.0-cp34-cp34m-manylinux1_x86_64.whl#md5=08a002aeffa20354aa5045eadb549361
  mv numpy-1.11.0-cp34-cp34m-manylinux1_x86_64.whl \
     numpy-1.11.0-cp34-none-linux_x86_64.whl
  pip3 install numpy-1.11.0-cp34-none-linux_x86_64.whl
  rm numpy-1.11.0-cp34-none-linux_x86_64.whl
fi

# Use pip to install scipy to get the latest version, instead of 0.13 through
# apt-get.
# pip install scipy==0.15.1
set +e
SCIPY_VERSION="0.15.1"
scipy_ver_flat=$(echo $SCIPY_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_scipy_ver=$(python -c "import scipy; print(scipy.__version__)")
local_scipy_ver_flat=$(echo $local_scipy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_scipy_ver_flat ]]; then
  local_scipy_ver_flat=0
fi
if (( $local_scipy_ver_flat < $scipy_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/00/0f/060ec52cb74dc8df1a7ef1a524173eb0bcd329110404869b392685cfc5c8/scipy-0.15.1-cp27-cp27mu-manylinux1_x86_64.whl#md5=aaac02e6535742ab02f2075129890714
  mv scipy-0.15.1-cp27-cp27mu-manylinux1_x86_64.whl \
     scipy-0.15.1-cp27-none-linux_x86_64.whl
  pip install scipy-0.15.1-cp27-none-linux_x86_64.whl
  rm scipy-0.15.1-cp27-none-linux_x86_64.whl
fi

# pip3 install scipy==0.15.1
set +e
local_scipy_ver=$(python3 -c "import scipy; print(scipy.__version__)")
local_scipy_ver_flat=$(echo $local_scipy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_scipy_ver_flat ]]; then
  local_scipy_ver_flat=0
fi
if (( $local_scipy_ver_flat < $scipy_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/56/c5/e0d36aaf719aa02ee3da19151045912e240d145586612e53b5eaa706e1db/scipy-0.15.1-cp34-cp34m-manylinux1_x86_64.whl#md5=d5243b0f9d85f4f4cb62514c82af93d4
  mv scipy-0.15.1-cp34-cp34m-manylinux1_x86_64.whl \
     scipy-0.15.1-cp34-cp34m-linux_x86_64.whl
  pip3 install scipy-0.15.1-cp34-cp34m-linux_x86_64.whl
  rm scipy-0.15.1-cp34-cp34m-linux_x86_64.whl
fi

# pip install sklearn
set +e
SKLEARN_VERSION="0.17.1"
sklearn_ver_flat=$(echo $SKLEARN_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_sklearn_ver=$(python -c "import sklearn; print(sklearn.__version__)")
local_sklearn_ver_flat=$(echo $local_sklearn_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_sklearn_ver_flat ]]; then
  local_sklearn_ver_flat=0
fi
if (( $local_sklearn_ver_flat < $sklearn_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/bf/80/06e77e5a682c46a3880ec487a5f9d910f5c8d919df9aca58052089687c7e/scikit_learn-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl#md5=337b91f502138ba7fd722803138f6dfd
  mv scikit_learn-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl \
     scikit_learn-0.17.1-cp27-none-linux_x86_64.whl
  pip install scikit_learn-0.17.1-cp27-none-linux_x86_64.whl
  rm scikit_learn-0.17.1-cp27-none-linux_x86_64.whl
fi

# pip3 install scikit-learn
set +e
local_sklearn_ver=$(python3 -c "import sklearn; print(sklearn.__version__)")
local_sklearn_ver_flat=$(echo $local_sklearn_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_sklearn_ver_flat ]]; then
  local_sklearn_ver_flat=0
fi
if (( $local_sklearn_ver_flat < $sklearn_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/7e/f1/1cc8a1ae2b4de89bff0981aee904ff05779c49a4c660fa38178f9772d3a7/scikit_learn-0.17.1-cp34-cp34m-manylinux1_x86_64.whl#md5=a722a7372b64ec9f7b49a2532d21372b
  mv scikit_learn-0.17.1-cp34-cp34m-manylinux1_x86_64.whl \
     scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl
  pip3 install scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl
  rm scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl
fi

set -e

# pandas required by tf.learn/inflow
pip install pandas==0.18.1
pip3 install pandas==0.18.1

# Benchmark tests require the following:
pip install psutil
pip3 install psutil
pip install py-cpuinfo
pip3 install py-cpuinfo

# pylint tests require the following:
pip install pylint
pip3 install pylint

# pep8 tests require the following:
pip install pep8
pip3 install pep8

# tf.mock require the following for python2:
pip install mock

pip install portpicker
pip3 install portpicker
