# Chris Riederer
# 2017-05-27

"""Use simulated annealing to move a point cloud to an X.

This is an implementation of the code described in the paper "Same Stats, 
Different Graph" by Justin Matejka and George Fitzmaurice, CHI 2017. Check out 
their work at https://www.autodeskresearch.com/publications/samestats
"""


import io
import math
import random

import imageio

import numpy as np
from numpy.random import normal
import pandas as pd
import pylab as pl
from shapely.geometry import Point, MultiLineString


################################################################################
## Utility functions

def get_x(points):
  return points[:, 0]

def get_y(points):
  return points[:, 1]

def get_point_mean_x(points):
  return sum(get_x(points)) / len(points)

def get_point_mean_y(points):
  return sum(get_y(points)) / len(points)

def get_point_sd_x(points):
  return np.array(get_x(points)).std()

def get_point_sd_y(points):
  return np.array(get_y(points)).std()

def get_stats(points):
  return [
      get_point_mean_x(points),
      get_point_mean_y(points),
      get_point_sd_x(points),
      get_point_sd_y(points),
  ]


################################################################################
## IO

def print_stats(points):
  print("%.6f\t%.6f\t%.6f\t%.6f" % tuple(get_stats(points)))


def write_points(points, fname):
  df = pd.DataFrame()
  df["x"] = get_x(points)
  df["y"] = get_y(points)
  df.to_csv(fname, index=False)


def read_target_points(fname):
  """Converts CSV filename of points coordinates to list our code can use.

  Takes the filename of a CSV with columns "x" and "y". Returns a list of tuples
  where each tuple is a start point and an end point. Since this is going for
  points and not line-segments, that tuple will just be repeats.
  """
  df = pd.read_csv(fname)
  point_list = df[["x", "y"]].to_records().tolist()
  point_list = [(x, y) for idx, x, y in point_list]
  point_list = [(point, point) for point in point_list]
  return point_list


def read_target_lines(fname):
  """Converts filename of CSV containing line coords to list our code can use.

  Takes the filename of a CSV with columns "x1", "y1", "x2", and "y2". 
  Returns a list of tuples where each tuple is a start point and an end point 
  for a line segment which can be input into Shapely.
  """
  df = pd.read_csv(fname)
  point_list = df[["x1", "y1", "x2", "y2"]].to_records().tolist()
  point_list = [((x1, y1), (x2, y2)) for idx, x1, y1, x2, y2 in point_list]
  return point_list


def plot_points(points, target, orig_points=None):
  if orig_points is not None:
    pl.scatter(orig_points[:, 0], orig_points[:, 1], 
               color="blue", s=50, alpha=0.3)
  x = points[:, 0]
  y = points[:, 1]
  # px, py = target.xy
  # pl.plot(px, py, alpha=0.5)
  pl.scatter(x, y, color="red")
  pl.xlim(0, 100)
  pl.ylim(0, 100)


def prep_gifwriter(gif_fname):
  if gif_fname is not None:
    try:
      import imageio
      return imageio.get_writer(gif_fname, mode='I')
    except ImportError:
      print("imageio not installed. Please install imageio to make gifs.")
      print("Continuing")
      return None
  else:
    gif_writer = None


def OutputInfo(i, current_ds, target, initial_ds, num_loops, fig_fname, 
               gif_writer, show_plots=True):
  print(i, end="\t")
  print("%.4f" % FitnessOverall(current_ds, target), sep="\t", end="\t")
  print_stats(current_ds)
  pl.clf()
  plot_points(current_ds, target, orig_points=initial_ds)
  if show_plots:
    pl.pause(0.000001)
    pl.show()

  if fig_fname is not None:
    this_name = "{fig_fname}_{step:0>{num_place}}.png".format(
        fig_fname=fig_fname, step=i, num_place=int(math.log10(num_loops)))
    pl.savefig(this_name)

  if gif_writer is not None:
    buf = io.BytesIO()
    pl.savefig(buf, format='png')
    buf.seek(0)
    gif_writer.append_data(imageio.imread(buf))


################################################################################
## Simulated Annealing code

def FitnessOverall(points, lines):
  """Fitness for our entire set of points"""
  return -sum([lines.distance(Point(x, y)) for x, y in points])


def Fitness(point, lines):
  """Fitness for one point"""
  return -lines.distance(Point(point[0], point[1]))


def Perturb(ds, target, temp, scale=0.1):
  """Perturb the dataset.
  Move one randomly select point from "ds" by a normal variable with standard
  deviation equal to scale. Accept the new point if it either improves the 
  overall fitness, or if a random number is above the current number, return it.
  Make 1000 attempts to change the dataset before returning original dataset. 
  Depending on the size of your dataset, you may want to modify scale.
  ds: dataset, n x 2 numpy array
  target: the Shapely object we're moving the points towards
  temp: current temperature, should be between 0 and 1
  scale: standard deviation of the normal variable by which we move the point
  """
  for i in range(1000):
    # Move a random point
    rand_idx = np.random.randint(len(ds))
    old_point = ds[rand_idx, :]
    new_point = old_point + normal(scale=scale, size= 2)

    if (temp > np.random.random() 
          or Fitness(new_point, target) > Fitness(old_point, target)):
      out = ds.copy()
      out[rand_idx, :] = new_point
      return out
  print("failed to pass")
  return ds


def constraint(old, new):
  """Constraint: first two decimal places remain the same"""
  return math.floor(old*100) == math.floor(new*100)


def IsErrorOk(test_ds, initial_ds):
  return (
    constraint(test_ds[:, 0].mean(), initial_ds[:, 0].mean())
    and constraint(test_ds[:, 1].mean(), initial_ds[:, 1].mean())
    and constraint(test_ds[:, 0].std(), initial_ds[:, 0].std())
    and constraint(test_ds[:, 1].std(), initial_ds[:, 1].std())
  )


def FindDataset(initial_ds, target, num_loops=2e5, step_size=100, 
                show_plots=True, fig_fname=None, gif_fname=None):
  """"""
  current_ds = initial_ds.copy()
  temperatures = np.logspace(np.log10(0.4), np.log10(0.1), num=num_loops)

  # Prep image display and GIF creation
  if show_plots:
    pl.figure()
    pl.show()
  gif_writer = prep_gifwriter(gif_fname)
  
  # Run everything
  for i, temp in enumerate(temperatures):
    test_ds = Perturb(current_ds, target, temp)  
    if IsErrorOk(test_ds, initial_ds):
      current_ds = test_ds

    if i % step_size == 0:
      # Update the user w/ current state. Make images and gifs.
      OutputInfo(i, current_ds, target, initial_ds, num_loops, fig_fname, 
                 gif_writer, show_plots=show_plots)

  return current_ds


def main():
  # Generate random data set
  np.random.seed(1337)
  points = np.random.random((182, 2)) * 100

  # Create target Polygon
  # coords = [
  #     ((15, 15), (15, 15)),
  #     ((15, 50), (15, 50)),
  #     ((15, 85), (15, 85)),
  #     ((50, 15), (50, 15)),
  #     ((50, 50), (50, 50)),
  #     ((50, 85), (50, 85)),
  #     ((85, 15), (85, 15)),
  #     ((85, 50), (85, 50)),
  #     ((85, 85), (85, 85)),
  # ]
  # coords = read_target_points("targets/grid.csv")
  coords = read_target_points("targets/Datasaurus_data.csv")
  # coords = read_target_lines("targets/box.csv")
  target = MultiLineString(coords)

  # Turn interactive mode on for plotting
  pl.ion()

  out_points = FindDataset(points, target, 2e5, step_size=1000, )
      # gif_fname="fig/gif/test.gif")
      # fig_fname="fig/grid/grid")
  print("done.")
  # write_points(current_ds, "grid_points.csv")

  pl.ioff()
  plot_points(out_points, target, orig_points=points)
  pl.show()


if __name__ == '__main__':
  main()
