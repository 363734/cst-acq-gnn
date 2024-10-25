#! /bin/bash
# Step 1: creation/loading of datasets (creating of the pickle file for each models)
# Step 1-2: aggregate the stats about the instances

datadir="../target/data"
scriptdir="../src"

python ${scriptdir}/models/benchmark_stats.py -d $datadir
