#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.7:   ChenCode3-riboProof.py
Created 2019  (nelson@physics.upenn.edu)

Supplement to Chen, Zuckerman, and Nelson "Stochastic Simulation to Visualize Gene 
Expression and Error Correction in Living Cells"
Description: create and display stochastic simulation of ribosome kinetic proofreading process
Writes kproof_realistic.npz or kproof_HN.npz for display by ChenCode4-kproofBackend.ipynb

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import datetime
import time
#%%
# Main code begins at MAIN below. 
# The following function is the Gillespie simulation engine.

def ribosim(ratesmat, kadd_C, kadd_W, chainlength):
    '''
    inputs:
        ratesmat[i,j] = rate to transition from state j to i [=0 on diagonal]
        kadd_C, kadd_W = rates of the amino acid incorporation steps
        chainlength = total number of amino acids to incorporate 
    outputs:
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

## initialize
    nstate = len(ratesmat)        # number of states, ratesmat defined with R matrix below
    Nreport = 101                 # report progress every Nreport amino acids
    incorpprobc = kadd_C/(ratesmat[0,3]) # incorporation rate for correct amino acid, see below for kadd_C
    incorpprobw = kadd_W/(ratesmat[0,4]) # incorporation rate for wrong amino acid, see below for kadd_W
    t = 0.        # current time
    states = [0]  # initially in state 0, empty ribosome 
    currentstate = 0
    ts = []     # array to store transition times
    chain = ""  # string to record amino acid chain
    Rtot = np.sum(ratesmat,axis=0) # sum of rates along columns, which is total rate to go from state j to the 4 other states
    propens = ratesmat.copy()  # make a scratch copy that we can modify
    for j in range(nstate):  # normalize rates to get probabilities of transitions
        propens[:,j]=ratesmat[:,j]/Rtot[j]
    cumu = np.cumsum(propens,axis=0) # cumulumative sum along columns, will use to determine which transition occurred 
    wrongs = 0 # initialize number of wrong incorporated AA
### perform the Gillespie simulation
    while len(chain) < chainlength: 
        if len(chain) > Nreport: # progress reporting
            print('*')
            Nreport = Nreport + 100
        decide = np.random.random(3) # 3 decisions need to be made independently
        # first choose time of the next transition:
        tstep = - np.log(decide[0])/Rtot[currentstate] # drawing transition time from an exponential distribution 
                                                       # exponential dist of waiting times comes from Poisson process
        t += tstep               # time at which we exit currentstate 
        ts = ts + [t]            # append transition time to array
        # next decide type of the next transition:
        newstate = np.nonzero(cumu[:,currentstate]>decide[1])[0][0] # first True entry is our next state
        if currentstate==3 and newstate==0: # check if tRNA.GDP is bound, if so, decide whether to incorporate
            if decide[2] < incorpprobc: 
                chain = chain + "*"
                states[-1] = -1*states[-1] # flag this state as actually incorporating an amino by making it negative
                print("*", len(chain))
        if currentstate==4 and newstate==0: # check if tRNA.GDP is bound, if so, decide whether to incorporate
            if decide[2] < incorpprobw: 
                chain = chain + "X"
                wrongs +=1
                states[-1] = -1*states[-1] # flag this state as actually incorporating an amino by making it negative
                print("X", len(chain))
        currentstate = newstate
        states = states + [currentstate] # add newstate to array of states
    return (chain, ts, states, wrongs)


#%% MAIN
# set up simulation parameters
    
chainlength = 25 # terminate when amino acid chain gets this long
numchains = 10    # how many amino acid chains to generate

simtype = "real" # determines model to simulate, HN for Hopfield-Ninio, Equil for equilibrium, real for realistic
# all rates below in 1/s, see Table 1 in text 
if simtype == "HN": 
    '''parameters for pure Hopfield-Ninio model. The only differences between correct
    and wrong amino acids are in the unbinding rates, and those differences are taken
    to be small to exaggerate the rate of wrong incorporations, for illustration.'''
    kadd_C = 0.01 # final incorporation step rate, C for correct, W for wrong
    phi_add = 1; kadd_W = kadd_C*phi_add
    
    # fixed parameters for ribosome cycle - correct amino acid; parameters based on in-vitro experimental measurements
    kc_on = 40    # k'_c
    kc_off = 0.5  # k_c
    lc_on = 1e-3   # l'_c very unlikely for GDP-bound tRNA-complex to bind 
    lc_off = 0.085  # l_c
    mhc = 0.01 # m'_c
    msc = 1e-3  # m_c very unlikely for for GTP synthesis to occur
    
    # fixed parameters for ribosome cycle - wrong amino acid
    phi1 =  1; kw_on = phi1*kc_on
    phim1 = 5; kw_off = phim1*kc_off  # W binds more weakly than C if phim1 > 1
    phim3 = 1; lw_on = phim3*lc_on
    phi2 =  1; mhw = phi2*mhc
    phim2 = 1; msw = phim2*msc
    phi3 =  phim1; lw_off = phi3*lc_off   # dictated from others by thermodynamic consistency, see Appendix A.1
    

elif simtype == "Equil": 
    '''no thermodynamic driving model'''
    kadd_C = 0.01 
    phi_add = 1; kadd_W = kadd_C*phi_add
    
    # fixed parameters for ribosome cycle - correct amino acid
    kc_on = 40    # k'_c
    kc_off = 0.5  # k_c
    lc_off = 0.085  # l_c
    mhc = 0.01 # m'_c
    
    # calculate lc_on and msc based on equilibrium thermodynamic consistency condition, see Eqn. 2 in Appendix A.1
    lc_on = np.sqrt((kc_on*mhc*lc_off)/(kc_off))
    msc = lc_on
    
    # fixed parameters for ribosome cycle - wrong amino acid
    phi1 =  1; kw_on = phi1*kc_on
    phim1 = 5; kw_off = phim1*kc_off  # W binds more weakly than C if phim1 > 1
    phim3 = 1; lw_on = phim3*lc_on
    phi2 =  1; mhw = phi2*mhc
    phim2 = 1; msw = phim2*msc
    phi3 =  phim1; lw_off = phi3*lc_off   # dictated from others by thermodynamic consistency, see Appendix A.1
    
elif simtype == "real": # use parameter values from Banerjee et al., www.pnas.org/cgi/content/short/1614838114
    '''parameters for realistic model. Forward and unbinding rates are now both different.'''
    kadd_C = 4.14 
    phi_add = 0.017; kadd_W = kadd_C*phi_add
    
    # fixed parameters for ribosome cycle - correct amino acid
    kc_on = 40    # k'_c
    kc_off = 0.5  # k_c
    lc_on = 1e-3   # l'_c very unlikely for GDP-bound tRNA-complex to bind  
    lc_off = 0.085  # l_c
    mhc = 25 # m'_c
    msc = 1e-3  # m_c very unlikely for for GTP synthesis to occur
    
    # fixed parameters for ribosome cycle - wrong amino acid
    phi1 =  0.68; kw_on = phi1*kc_on
    phim1 = 94; kw_off = phim1*kc_off  
    phi3 =  7.9; lw_off = phi3*lc_off
    phi2 =  0.048; mhw = phi2*mhc
    phim2 = 1; msw = phim2*msc
    phim3 = phi1*phi2*phi3/(phim1*phim2); lw_on = phim3*lc_on # dictated from others by thermodynamic consistency, see Appendix A.1
    
# set up ratesmat matrix, which is called R below    
'''    states: 
            0=ribosome A site empty
            1=binds C.GTP (right)
            2=binds W.GTP (wrong)
            3=binds C.GDP
            4=binds W.GDP
'''
R = np.zeros((5,5)) # R[i,j]=rate to transition from state j to i, or 0 if j=i
R[0,0] = 0.
R[1,0] = kc_on
R[2,0] = kw_on
R[3,0] = lc_on
R[4,0] = lw_on
#
R[0,1] = kc_off
R[1,1] = 0.
R[2,1] = 0.
R[3,1] = mhc
R[4,1] = 0.
#
R[0,2] = kw_off
R[1,2] = 0.
R[2,2] = 0.
R[3,2] = 0.
R[4,2] = mhw
#
R[0,3] = lc_off + kadd_C # total rate to unbind or incorporate
R[1,3] = msc
R[2,3] = 0.
R[3,3] = 0.
R[4,3] = 0.
#
R[0,4] = lw_off + kadd_W # total rate to unbind or incorporate
R[1,4] = 0.
R[2,4] = msw
R[3,4] = 0.
R[4,4] = 0.

#%% run Gillespie simulation with parameters defined above
print (R)
chains = [] # placeholder for a list with simulated chains
ts = []     # placeholder for a list of transition times for each chain
states = [] # placeholder for a list of states for each chain
for i in range(1,numchains+1):
    print("chain ", i)
    chainM, tsM, statesM, wrongs = ribosim(R, kadd_C, kadd_W, chainlength) # use M for each iteration, no M for multiple chains
    chains = chains + [chainM]
    ts = ts + [tsM]
    states = states + [statesM]
    print(len(tsM)," transitions; ", wrongs," errors")
chains = np.array(chains)
ts = np.array(ts)
states = np.array(states)
now = datetime.datetime.now() # date when the simulation was done
np.savez('kproof_'+str(simtype)+time.strftime("_%m%d%Y"), chains=chains, ts=ts, states=states, now=now, simtype=simtype, R=R) # save simulation data
#%% view data from a previous simulation
#data = np.load('kproof_real_08252018.npz') # example data file name shown here
#print("Data generated ", data['now']," in simulation type ", data['simtype'])
#ts = data['ts'][0]   # pulls up data for chain 0, change number to change which chain data is loaded       
#states = data['states'][0]  
#chains = data['chains'][0]
#R = data['R']