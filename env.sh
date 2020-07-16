#! /usr/bin/env bash

export RUN_DB_PATH="$HOME/runs/sac"
export RUN_IMAGE_BUILD_PATH="$HOME/jax-rl/"
export RUN_DOCKERFILE_PATH="$RUN_IMAGE_BUILD_PATH/Dockerfile"
export RUN_IMAGE='sac'
export RUN_CONFIG_SCRIPT='config_script.py'
export RUN_CONFIG_SCRIPT_INTERPRETER='python3'
export RUN_CONFIG_SCRIPT_INTERPRETER_ARGS='-c'
export DOCKER_RUN_COMMAND='docker run -d --rm --gpus all -it'
