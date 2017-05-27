import pandas as pd
import pylab as pl
import numpy as np
import simanneal

import random

np.random.seed(1337)


foo = 2*np.random.random(1000) - 1
foo = foo.cumsum()

# pl.plot(foo)
# pl.show()
print(np.argmax(foo))


class MaxFinder(simanneal.Annealer):
  """Test annealer with a simple maximization problem
  """

  # pass extra data (the array) into the constructor
  def __init__(self, state, arr):
    self.arr = arr
    self.step = 0
    super(MaxFinder, self).__init__(state)  # important!

  def move(self):
    """Pick random neighbor"""
    # Note: we do a "wrap around" thing here to avoid boundaries.
    self.step = self.step+1
    if self.step % 100 == 0:
      print(self.state[0])
      # print(dir(self.state[0]))
    diff = random.randint(-10, 10) 
    next_ind = (self.state[0] + diff) % len(self.arr)
    self.state = [next_ind]

  def energy(self):
    """Calculates the length of the route."""
    return -self.arr[self.state[0]]


prob = MaxFinder([0], foo)
prob.steps = 10000
# since our state is just a list, slice is the fastest way to copy
# prob.copy_strategy = "slice"
state, e = prob.anneal()
print(state, e)
