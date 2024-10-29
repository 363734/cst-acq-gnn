#! /bin/bash

module load Python/3.11.3-GCCcore-12.3.0
module load virtualenv

virtualenv venv --python=python3.11

python -m pip install -r requirements.txt
