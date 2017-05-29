# Chris Riederer
# 2017-05-27

"""Use simulated annealing to move a point cloud to an X"""

import math
import random

import numpy as np
from numpy.random import normal
import pandas as pd
import pylab as pl
import simanneal
from shapely.geometry import LinearRing, Polygon, Point

# Generate data set
np.random.seed(1337)
points = np.random.random((182, 2)) * 100
# points = [Point(p) for p in points]

# Create target Polygon
poly = Polygon([(5, 20), (5, 80), (95, 80), (95, 20)])
poly_ext = LinearRing(poly.exterior.coords)
px, py = poly_ext.xy

# Turn interactive mode on for plotting
pl.ion()

def plot_points(points, poly_ext, orig_points=False):
  if orig_points:
    orig_x = [p.x for p in orig_points]
    orig_y = [p.y for p in orig_points]
    pl.scatter(orig_x, orig_y, color="red", s=50)
  x = [p.x for p in points]
  y = [p.y for p in points]
  pl.scatter(x, y)
  pl.plot(px, py)
  pl.xlim(0, 100)
  pl.ylim(0, 100)
  pl.pause(0.0001)
  pl.show()

def plot_pointsNP(points, poly_ext, orig_points=False):
  if orig_points:
    pl.scatter(orig_points[:, 0], orig_points[:, 1], color="blue", s=50, alpha=0.7)
  x = points[:, 0]
  y = points[:, 1]
  pl.scatter(x, y)
  px, py = poly_ext.xy
  pl.plot(px, py)
  pl.xlim(0, 100)
  pl.ylim(0, 100)
  pl.pause(0.0001)
  pl.show()

def points_from_numpy(arr):
  return []

# Utility functions
def get_x(points):
  return points[:, 0]  # [p.x for p in points]
def get_y(points):
  return points[:, 1]  #[p.y for p in points]
def get_point_mean_x(points):
  return sum([p.x for p in points]) / len(points)
def get_point_mean_y(points):
  return sum([p.y for p in points]) / len(points)
def get_point_sd_x(points):
  return np.array([p.x for p in points]).std()
def get_point_sd_y(points):
  return np.array([p.y for p in points]).std()
def get_stats(points):
  return [
      get_point_mean_x(points),
      get_point_mean_y(points),
      get_point_sd_x(points),
      get_point_sd_y(points),
  ]
def print_stats(points):
  print("%.6f\t%.6f\t%.6f\t%.6f" % tuple(get_stats(points)))

def write_points(points, fname):
  df = pd.DataFrame()
  df["x"] = [p.x for p in points]
  df["y"] = [p.y for p in points]
  df.to_csv(fname, index=False)


def TestAvgDifference(points, scale, num_loops=20):
  orig_stats = np.array(get_stats(points))
  new_stats = np.array([get_stats(MoveRandomPoints(points, scale))
                          for _ in range(num_loops)])
  return np.abs(new_stats - orig_stats).mean(axis=0)

def TestErrorPassRate(points, scale, num_loops=100):
  orig_stats = np.array(get_stats(points))
  passes = np.array([IsErrorOk(points, MoveRandomPoints(points, scale))
                     for _ in range(num_loops)])
  return passes.mean()

def Fit(points):
  return -sum([poly_ext.distance(p) for p in points])

def FitOverall(points):
  return -sum([poly_ext.distance(Point(x, y)) for x, y in points])

def Fit1(point):
  return -poly_ext.distance(Point(point[0], point[1]))


def PerturbNP(ds, temp):
  # loop: ?
  for i in range(1000):
    # Move a random point
    rand_idx = np.random.randint(len(ds))

    old_point = ds[rand_idx, :]
    new_point = old_point + normal(scale=scale, size= 2)

    if temp > np.random.random() or Fit1(new_point) > Fit1(old_point):
      ds[rand_idx, :] = new_point
      return ds
  print("failed to pass")
  return ds


def Perturb(ds, temp):
  # loop: ?
  for i in range(1000):
    test = MoveRandomPoints(ds)
    if temp > np.random.random() or Fit(test) > Fit(ds):
      return test
  print("failed to pass")
  return ds

def constraint(old, new):
  """Constraint: first two decimal places remain the same"""
  return math.floor(old*100) == math.floor(new*100)


def IsErrorOkNP(test_ds, initial_ds):
  x_new = np.array(get_x(test_ds))
  y_new = np.array(get_y(test_ds))
  x_old = np.array(get_x(initial_ds))
  y_old = np.array(get_y(initial_ds))
  return (
    constraint(x_new.mean(), x_old.mean())
    and constraint(y_new.mean(), y_old.mean())
    and constraint(x_new.std(), x_old.std())
    and constraint(y_new.std(), y_old.std())
  )

def IsErrorOkNP2(test_ds, initial_ds):
  return (
    constraint(test_ds[:, 0].mean(), initial_ds[:, 0].mean())
    and constraint(test_ds[:, 1].mean(), initial_ds[:, 1].mean())
    and constraint(test_ds[:, 0].std(), initial_ds[:, 0].std())
    and constraint(test_ds[:, 1].std(), initial_ds[:, 1].std())
  )

def IsErrorOkOrig(test_ds, initial_ds):
  return (
    constraint(get_point_mean_x(test_ds), get_point_mean_x(initial_ds))
    and constraint(get_point_mean_y(test_ds), get_point_mean_y(initial_ds))
    and constraint(get_point_sd_x(test_ds), get_point_sd_x(initial_ds))
    and constraint(get_point_sd_y(test_ds), get_point_sd_y(initial_ds))
  )

def MoveRandomPoints(ds, scale=0.019):
  # Move between 1 and 10 points
  num_points = np.random.randint(1, 30)

  random.shuffle(ds)
  update_points = ds[:num_points]
  update_points = [Point(p.x + normal(scale=scale), p.y + normal(scale=scale))
                   for p in update_points]
  new_points = update_points + ds[num_points:]
  return new_points


initial_ds = points
NUM_LOOPS = 100000
current_ds = initial_ds
temperatures = np.logspace(np.log10(0.4), np.log10(0.1), num=NUM_LOOPS)
pl.figure()
pl.show()

for i, temp in enumerate(temperatures):
  # test_ds = Perturb(current_ds, temp)
  test_ds = PerturbNP(current_ds, temp)  
  if IsErrorOkNP2(test_ds, initial_ds):
    current_ds = test_ds

  if i % 100 == 0:
    # print(i, "%.4f" % Fit(current_ds), end="\t")
    print(i, "%.4f" % FitOverall(current_ds), end="\t")
    print_stats(current_ds)
    pl.clf()
    plot_pointsNP(current_ds, poly_ext, orig_points=initial_ds)
    # plot_points(current_ds, poly_ext, orig_points=initial_ds)

write_points(current_ds, "box_points.csv")


