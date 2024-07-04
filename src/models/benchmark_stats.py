import argparse
import os
from typing import List

from src.utils.memoization.memoization import j_save

from src.models.model import Model


STATFILE = "stats.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, required=True, help="benchmark directory")
    args = parser.parse_args()
    return args


def get_stats(models: List[Model]):
    stats = {}
    for m in models:
        stats[m.name] = m.get_stats()
    return stats


if __name__ == "__main__":
    # Setup
    args = parse_args()

    models = []
    # fetch all the models
    for subdir, dirs, files in os.walk(args.directory):
        if len(files) > 0:
            spl = os.path.split(subdir)
            name = spl[1]
            spl = os.path.split(spl[0])
            family = spl[1]
            if args.directory == spl[0]:
                models.append(Model(name, family, args.directory))

    stats = get_stats(models)
    j_save(os.path.join(args.directory, STATFILE), stats)
    print(models)
