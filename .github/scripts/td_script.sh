#!/bin/bash

export TORCHRL_BUILD_VERSION=0.5.0

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
