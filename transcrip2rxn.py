# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:21:12 2015
Author: pcn 
Python:3.4.2
Description: birthdeath/transcrip2rxn.py two-channel Gillespie algorithm (birth/death process)
This code defines a function to be called by viewTransc.py
With thanks to Mark Kittisopikul and Daniel Andor
For details see "Physical models of living systems" by Philip Nelson
"""
import numpy as np#, matplotlib.pyplot as plt
def transcrip2rxn(lini, T, ks):
    ''' inputs:
    lini = initial number of mRNA
    T = total time to run
    outputs:
    ts = times at which x changed
    ls = running values of x just after those times
    ks = rate constants in 1/minute for synthesis and degradation'''
#%% Parameters:
    stoich = np.array([0, 1]);     # reaction orders
#%% initialize
    t = 0; treport=100    # current time
    x = lini  # current mRNA population
    ts = [t] # histories
    ls = [x]
    #%%
    while t<T:
        if t>treport:
            print(t)
            treport=treport+100
        a = (x**stoich) * ks;             # propensities
        atot = np.sum(a);                 # total rate for anything to happen
        t = t - np.log(np.random.random())/atot
# make birth/death decision based on the relative propensity:
        mu = 1 - 2*(a[0]/atot < np.random.random())
        x = x + mu
        ts = ts + [t]
        ls = ls + [x]
    return (np.array(ts), np.array(ls))