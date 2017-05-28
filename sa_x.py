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


np.random.seed(1337)


points = np.random.random((182, 2)) * 100

# pl.show()

mean = points.mean(axis=0)
mean_x = mean[0]
mean_y = mean[1]
sd_x = points[:, 0].std()
sd_y = points[:, 1].std()

pts = [Point(p) for p in points]

poly = Polygon([(5, 20), (5, 80), (95, 80), (95, 20)])
# px, py = poly.exterior.xy
poly_ext = LinearRing(poly.exterior.coords)
px, py = poly_ext.xy

pl.ion()

# pl.scatter(points[:, 0], points[:, 1])
# pl.plot(px, py)
# pl.show()


class ShapeFinder(simanneal.Annealer):
  """Simulated annealing to find point clouds that look like shapes with 
  pre-specified summary statistics.
  """

  def __init__(self, points, poly):
    """Points should be an n x 2 numpy array. 
    poly is a Shapely polygon
    """
    self.step = 0

    pl.figure()

    mean = points.mean(axis=0)
    self.mean_x = mean[0]
    self.mean_y = mean[1]
    self.sd_x = points[:, 0].std()
    self.sd_y = points[:, 1].std()

    print(self.mean_x, self.mean_y, self.sd_x, self.sd_y)

    self.poly = poly

    self.points = points
    state = [Point(p) for p in points]

    self.step = 0
    super(ShapeFinder, self).__init__(state)  # important!

  @staticmethod
  def constraint(old, new):
    """Constraint: first two decimal places remain the same"""
    return math.floor(old*100) == math.floor(new*100)

  def check_all_constraints(self):
    return (
      self.constraint(self.mean_x, self.points.mean(axis=0)[0])
      and self.constraint(self.mean_y, self.points.mean(axis=0)[1])
      and self.constraint(self.sd_x, self.points[:, 0].std())
      and self.constraint(self.sd_y, self.points[:, 1].std())
    )    

  def update(self, *args, **kwargs):
      """Wrapper for internal update.
      If you override the self.update method,
      you can chose to call the self.default_update method
      from your own Annealer.
      """
      if self.step % 1000 == 0:
        # print("showing fig...")
        # pl.figure()
        pl.clf()
        pl.scatter(self.points[:, 0], self.points[:, 1])
        px, py = self.poly.xy
        pl.plot(px, py)
        pl.pause(0.0001)
        pl.show()
      self.default_update(*args, **kwargs)

  def move(self):
    """Pick random neighbor"""
    self.step += 1

    # Pick which point to move
    # idx = np.random.randint(len(self.state))
    # old_point_sp = self.state[idx]
    # old_point_np = self.points[idx].copy()

    old_state = self.state
    old_points = self.points.copy()

    # Perturb
    self.points = (old_points + 
                    (np.random.normal(scale=0.005, size=old_points.shape) *
                      (np.random.rand(*old_points.shape)<.1).astype(int)))
    self.state = [Point(p) for p in self.points]
    # new_x = old_point_np[0] + np.random.normal(scale=0.005)
    # new_y = old_point_np[1] + np.random.normal(scale=0.005)
    # self.points[idx] = [new_x, new_y]
    # self.state[idx] = Point(new_x, new_y)
    
    # if self.step % 1000 == 0:
    #   print(old_point_np, [new_x, new_y])

    # Check constraints, revert to old state if not met
    if not self.check_all_constraints():
      # print("nope")
      self.points = old_points
      self.state = old_state

      # print(self.mean_x, self.mean_y, self.sd_x, self.sd_y)
      # print(self.points.mean(axis=0)[0], self.points.mean(axis=0)[1], 
      #       self.points[:, 0].std(), self.points[:, 1].std())

    #   print()
      # pl.scatter(self.points[:, 0], self.points[:, 1])
      # px, py = self.poly.xy
      # px, py = self.poly.exterior.xy
      # pl.plot(px, py)
      # pl.show()

  def energy(self):
    """Calculates the length of the route."""
    return sum([self.poly.distance(p) for p in self.state])


prob = ShapeFinder(points, poly_ext)
prob.steps = 200000
# since our state is just a list, slice is the fastest way to copy
# prob.copy_strategy = "slice"
state, e = prob.anneal()
# print(state, e)

print((prob.points == points).all())

# px, py = self.poly_ext.xy
x = [p.x for p in state]
y = [p.y for p in state]

pl.ioff()

pl.scatter(x, y)
pl.plot(px, py)
pl.show()

