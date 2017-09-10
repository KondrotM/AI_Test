# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:14:00 2017

@author: ashbeck
"""
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)

lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked = True, color=['r','b'])