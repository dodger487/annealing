import numpy as np
import pandas as pd
import pylab as pl

pl.ion()


points = np.random.random((100, 2)) * 100

pl.figure()
pl.scatter(points[:, 0], points[:, 1])
pl.pause(0.0001)
# pl.plot(px, py)
pl.show()

print("hello")
