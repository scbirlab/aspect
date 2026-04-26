#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"

TEST_FEAT="mwt mwt@rename_mwt mwt:log@log_mwt smiles:descriptors-2d:log@log_desc_2d CollectionName:hash CollectionName:hash(ndim=8,normalize=true)@collection_hash"

aspect serialize \
    --features $TEST_FEAT \
    --output "$OUTPUT_DIR"/serialized/out.as

HF_HOME="$CACHE" aspect featurize \
    "$TRAIN" \
    --start 100 \
    --end 200 \
    --features $TEST_FEAT \
    --output "$OUTPUT_DIR"/featurized/out.csv

HF_HOME="$CACHE" aspect featurize \
    "$TRAIN" \
    --start 100 \
    --end 200 \
    --config "$OUTPUT_DIR"/serialized/out.as \
    --checkpoint "$OUTPUT_DIR"/featurized/out.as \
    --output "$OUTPUT_DIR"/featurized/out-from-config.csv

if [ "$(diff "$OUTPUT_DIR"/featurized/out-from-config.csv "$OUTPUT_DIR"/featurized/out.csv | wc -l)" -gt 0 ]
then
    exit 1
fi
