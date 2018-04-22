#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:04:03 2018

@author: Chandra Sekhar Ravuri
"""

###include
# optimization with dictionary

from cvxopt.modeling import variable, op, matrix
import numpy as np

np.random.seed(10)

eps = 10

D_device = np.random.rand(500,300) # dictionary
X = np.random.randint(1,20,(1,300))


a = variable(D_device.shape[0],name='Sparce code')

#si = variable(name='Dictionary')
#X = variable(name='Aggregate')


#const = ( a*D_device == X )


#abc = op(sum(abs(a)),[const])

#abc.solve()
