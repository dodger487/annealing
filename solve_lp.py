

import math

import numpy as np

from gurobipy import *

np.random.seed(1337)
points = np.random.random((182, 2)) * 100

coords = [
    (15, 15),
    (15, 50),
    (15, 85),
    (50, 15),
    (50, 50),
    (50, 85),
    (85, 15),
    (85, 50),
    (85, 85),
]

def get_x(points):
  return points[:, 0]  # [p.x for p in points]
def get_y(points):
  return points[:, 1]  #[p.y for p in points]
def get_point_mean_x(points):
  return sum(get_x(points)) / len(points)
def get_point_mean_y(points):
  return sum(get_y(points)) / len(points)
def get_point_sd_x(points):
  return np.array(get_x(points)).std()
def get_point_sd_y(points):
  return np.array(get_y(points)).std()


model = Model("same_stats")


num_points = len(points)
xs, ys = [], []
for i in range(num_points):
  this_x = model.addVar(name="x" + str(i))
  xs.append(this_x)
  this_y = model.addVar(name="y" + str(i))
  ys.append(this_y)

# Add constraints for mean
x_mean_const = get_point_mean_x(points)
x_mean = model.addVar(
  name="x_mean", 
  lb=math.floor(get_point_mean_x(points)*100)/100,
  ub=math.ceil(get_point_mean_x(points)*100)/100,
)
model.addConstr(quicksum(xs) / num_points == x_mean)
y_mean = model.addVar(
  name="y_mean", 
  lb=math.floor(get_point_mean_y(points)*100)/100,
  ub=math.ceil(get_point_mean_y(points)*100)/100,
)
model.addConstr(quicksum(ys) / num_points == y_mean)

# Add constraints for std
# x_std = model.addVar(
#   name="x_std", 
#   lb=math.floor(get_point_sd_x(points)*100)/100,
#   ub=math.ceil(get_point_sd_x(points)*100)/100,
# )
model.addQConstr(quicksum([(xi-x_mean_const)*(xi-x_mean_const) for xi in xs]) / num_points <= math.ceil(get_point_sd_x(points)*100)/100)
# model.addQConstr(quicksum([(xi-x_mean_const)*(xi-x_mean_const) for xi in xs]) / num_points >= math.floor(get_point_sd_x(points)*100)/100)
# model.addQConstr(xs[0]*xs[0] <= x_std)
# y_std = model.addVar(
#   name="y_std", 
#   lb=math.floor(get_point_sd_y(points)*100)/100,
#   ub=math.ceil(get_point_sd_y(points)*100)/100,
# )

model.update()

all_dists = []
for this_x, this_y in zip(xs, ys):
  this_x_dists = []
  this_y_dists = []
  for i, (x_goal, y_goal) in enumerate(coords):
    name = "_dist" + str(i)
    x_name = this_x.getAttr("VarName") + name
    x_dist = model.addVar(name=x_name)
    # model.addConstr(x_dist == abs(x_goal - this_x))
    z = model.addVar(lb=-GRB.INFINITY)
    model.addConstr(z == x_goal - this_x)
    model.addGenConstrAbs(x_dist, z)
    this_x_dists.append(x_dist)

    y_name = this_y.getAttr("VarName") + name
    y_dist = model.addVar(name=y_name)
    # model.addConstr(y_dist == abs(y_goal - this_y))
    z = model.addVar(lb=-GRB.INFINITY)
    model.addConstr(z == y_goal - this_y)
    model.addGenConstrAbs(y_dist, z)
    this_y_dists.append(y_dist)

  min_x_dist = model.addVar(name=this_x.getAttr("VarName") + "_mindist")
  model.addGenConstrMin(min_x_dist, this_x_dists)
  min_y_dist = model.addVar(name=this_y.getAttr("VarName") + "_mindist")
  model.addGenConstrMin(min_y_dist, this_y_dists)
  
  # all_dists.append(min_x_dist)
  # all_dists.append(min_y_dist)
  euclid_dist = model.addVar(name="total_dist_" + str(i), lb=0)
  model.addConstr(euclid_dist*euclid_dist >= min_x_dist*min_x_dist + min_y_dist*min_y_dist)
  all_dists.append(euclid_dist)

model.setObjective(quicksum(all_dists), GRB.MINIMIZE)
# model.params.BestObjStop = 15
model.optimize()

model_vars = model.getVars()
out_points = [v for v in model_vars if "_" not in v.varName]
x_out = [v.x for v in out_points if "x" in v.varName]
y_out = [v.x for v in out_points if "y" in v.varName]

import pandas as pd
df = pd.DataFrame()
df["x"] = x_out
df["y"] = y_out
df.to_csv("tmp.csv")

import pylab as pl
pl.scatter(get_x(points), get_y(points), color="blue")
pl.scatter(x_out, y_out, color="red")
pl.show()

# for v in model.getVars():
#   if "dist" in v.varName:
#     continue
#   print('%s %g' % (v.varName, v.x))
print('Obj: %g' % model.objVal)
# Loss function:
# For each point, constraint 

# Constraints:
# mean x ~= original mean x
# mean y ~= original mean y
# std x ~= original std x
# std y ~= original std y



# sum(x) / 182 <= mean_x ceiling
# sum(x) / 182 >= mean_x floor