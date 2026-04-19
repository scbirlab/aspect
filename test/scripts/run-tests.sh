#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"

HF_HOME="$CACHE" aspect --help

# Import sanity check
python -c "
from aspect.pipeline.data import DataPipeline, ColumnPipeline
from aspect.serializing import Preprocessor
from aspect.pipeline.io import AutoDataset
print('imports OK')
"

# Doctests
python -m pytest --doctest-modules \
    aspect/serializing.py \
    aspect/functions.py \
    aspect/pipeline/io.py \
    aspect/pipeline/data.py \
    -v

# Unit tests
python -m pytest test/test_pipeline.py -v
