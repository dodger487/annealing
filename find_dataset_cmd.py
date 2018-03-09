
# Chris Riederer
# 2017-06-04

"""Command line access to code for finding different graphs with same stats!

This is an implementation of the code described in the paper "Same Stats, 
Different Graph" by Justin Matejka and George Fitzmaurice, CHI 2017. Check out 
their work at https://www.autodeskresearch.com/publications/samestats
"""

import argparse
import pandas as pd

desc = ("Command line access to code for finding different graphs with same "
        "stats."
       )

parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "input_fname", type=str, 
    help="Filename of input points. File should be CSV with columns x and y.")
parser.add_argument(
    "--num_loops", default=2e5, type=int, help="Number of loops to run for.")
parser.add_argument(
    "--step_size", default=100, type=int, 
    help="Print output / save an image / make GIF frame ever step_size loops.")
parser.add_argument(
    "--gif_fname", default=None, type=str,
    help="Filename in which to save a GIF animation image of the annealing.")

args = parser.parse_args()


import numpy as np
import pylab as pl
from shapely.geometry import Point, MultiLineString

from find_dataset import FindDataset


points = pd.read_csv(args.input_fname)
points = np.array([[x, y] for x, y in zip(points.x, points.y)])

# Create target Polygon
coords = [
    ((15, 15), (15, 15)),
    ((15, 50), (15, 50)),
    ((15, 85), (15, 85)),
    ((50, 15), (50, 15)),
    ((50, 50), (50, 50)),
    ((50, 85), (50, 85)),
    ((85, 15), (85, 15)),
    ((85, 50), (85, 50)),
    ((85, 85), (85, 85)),
]
target = MultiLineString(coords)


pl.ion()

out_points = FindDataset(
    points, 
    target, 
    num_loops=args.num_loops, 
    step_size=args.step_size, 
    gif_fname=args.gif_fname,
)

print("done")

pl.ioff()
pl.show()

