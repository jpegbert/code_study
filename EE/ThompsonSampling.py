import numpy as np
import pymc


"""
Thompson sampling算法python实现
"""

successes = 10
totals = 100
np.argmax(pymc.rbeta(1 + successes, 1 + totals - successes))

