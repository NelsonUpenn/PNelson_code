# -*- coding: utf-8 -*-
"""
Updated 5/2018
Author: pcn Python:3.6
Description: ribosim.py, kinetic proofreading engine called by riboProof.py
Inspired by Dan Zuckerman notes:
see http://physicallensonthecell.org/cell-biology-phenomena/active-kinetic-proofreading
which themselves are a simplification of Hopfield PNAS 1974

"""
import numpy as np#, matplotlib.pyplot as plt
def ribosim(ratesmat, incorprate, chainlength):
    '''function called by riboProof.py
    inputs:
        ratesmat[i,j] = rate to transition from j to i [=0 on diagonal]
        incorprate = rate of the incorporation step
        chainlength = total number of amino acids to incorporate
    outputs:
        frac = fraction of time spent correctly loaded
        ts = times at which transitions occurred
        chain = amino acid chain (of length chainlength)
        states = list of states occupied during each dwell. states[i] is state exited at ts[i]. 
                 states[i+1] is state entered at ts[i].
    states: 
            0=ribosome A site empty
            1=binds C.GTP (correct)
            2=binds W.GTP (wrong)
            3=binds C.GDP (or -3 means that when this state exits, C amino acid is incorporated)
            4=binds W.GDP (or -4 means that when this state exits, W amino acid is incorporated)
'''
#%% Parameters:
    nstate = len(ratesmat)        # number of states
    Nreport = 101                 # how often to report progress
#%% initialize
    incorpprobc = incorprate/(incorprate+ratesmat[0,3]) # correct
    incorpprobw = incorprate/(incorprate+ratesmat[0,4]) # wrong
    t = 0.        # current time
    states = [0]  # initially in state 0
    currentstate = 0
    ts = []     # transition times
    chain = ""
    Rtot = np.sum(ratesmat,axis=0)
    propens = ratesmat.copy()  # make a scratch copy that we can modify
    for j in range(nstate):  # normalize rates to get probabilities of transitions
        propens[:,j]=ratesmat[:,j]/Rtot[j]
    cumu = np.cumsum(propens,axis=0)
    wrongs = 0 # number of wrong incorporated AA
#%% perform the Gillespie simulation
    while len(chain) < chainlength: 
        if len(chain) > Nreport: # progress reporting
            print('*')
            Nreport = Nreport + 100
        decide = np.random.random(3) # 3 decisions need to be made independently
        # first choose time of the next transition:
        tstep = - np.log(decide[0])/Rtot[currentstate]
        t += tstep               # time at which we exit currentstate 
        ts = ts + [t]
        # next decide type of the next transition:
        newstate = np.nonzero(cumu[:,currentstate]>decide[1])[0][0] # first True entry is our next state
        if currentstate==3 and newstate==0: # as we exit state 3, update chain
            if decide[2] < incorpprobc: 
                chain = chain + "*"
                states[-1] = -1*states[-1] # flag this state as actually incorporating an amino
                print("*")
        if currentstate==4 and newstate==0: # as we exit state 4, update chain
            if decide[2] < incorpprobw: 
                chain = chain + "X"
                wrongs +=1
                states[-1] = -1*states[-1] # flag this state as actually incorporating an amino
                print("X")
        currentstate = newstate
        states = states + [currentstate]
    return (chain, ts, states, wrongs)
