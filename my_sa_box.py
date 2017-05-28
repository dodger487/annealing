# Chris Riederer
# 2017-05-27

"""Use simulated annealing to move a point cloud to an X"""

import math
import random

import numpy as np
import pandas as pd
import pylab as pl
import simanneal
from shapely.geometry import LinearRing, Polygon, Point

# Generate data set
np.random.seed(1337)
points = np.random.random((182, 2)) * 100
points = [Point(p) for p in points]

# Create target Polygon
poly = Polygon([(5, 20), (5, 80), (95, 80), (95, 20)])
poly_ext = LinearRing(poly.exterior.coords)
px, py = poly_ext.xy

# Turn interactive mode on for plotting
pl.ion()

# Utility functions
def get_point_mean_x(points):
  return sum([p.x for p in points]) / len(points)
def get_point_mean_y(points):
  return sum([p.y for p in points]) / len(points)
def get_point_sd_x(points):
  return np.array([p.x for p in points]).std()
def get_point_sd_y(points):
  return np.array([p.y for p in points]).std()


def Perturb(ds, temp):
  # loop: ?
  for i in range(10):
    test = MoveRandomPoints(ds)
    if Fit(test) > Fit(ds) or temp > np.random.random():
      return test

def IsErrorOk(test_df, initial_ds):
  return (
    get_point_mean_x(test_ds) == get_point_mean_x(initial_ds)
    and get_point_mean_y(test_ds) == get_point_mean_y(initial_ds)
    and get_point_sd_x(test_ds) == get_point_sd_x(initial_ds)
    and get_point_sd_y(test_ds) == get_point_sd_y(initial_ds)
  )

def MoveRandomPoints(ds):
  num_points

initial_ds = pts


NUM_LOOPS = 20000
current_ds = initial_ds
for i in range(NUM_LOOPS):
  test_ds = Perturb(current_ds, temp)
  if IsErrorOk(test_ds, initial_ds):
    current_ds = test_ds




