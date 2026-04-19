#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"

HF_HOME="$CACHE" aspect --help
