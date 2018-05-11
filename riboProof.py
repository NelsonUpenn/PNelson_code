#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9 May 2018

@author: Philip Nelson (Python 3.6)
Description: riboProof2.py  simulate kinetic proofreading
Inspired by Dan Zuckerman notes:
see http://physicallensonthecell.org/cell-biology-phenomena/active-kinetic-proofreading
which themselves are a simplification of Hopfield PNAS 1974

Calls ribosim2.py, so run that first to define function ribosim. See its docstring.
Writes kproof.npz for display by kproof_backend.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# adjustable parameters
bias = 4.        # ratio of off-rates for wrong/right, = 1/f_0
conc_GTP = 1e-3  # in M; use 1e-3 (or 1e-6, closer to equil)
conc_GDP = 1e-6  # in M; use 1e-6 (or 1000e-6, closer to equil)
conc_Pi = 1e-6   # in M; use 1e-6  (or 1000e-6, closer to equil)
chainlength = 20 # terminate when chain gets this long
numchains = 1    # how many chains to generate

# fixed parameters - loading tRNA complexes with GTP
conc_C = 1e-4          # concentration in M; C means "correct"
conc_W = conc_C        # in M; W means "wrong"
K_D = 4.9e5            # in M, equil constant for GDP+Pi <--> GTP
kon = 1e8 # in 1/(M s), other on-rates are expressed as multiples of this base value
koff = 100 # in 1/s, other off-rates are expressed as multiples of this base value
incorprate = 9e-2*koff # in 1/s, final incorporation step rate; realistic value is  ~ 1e-3*koff
kgtp_on = kon          # g'_t
kgtp_off = koff        # g_t
kgdp_offN = 10*koff    # g_d. This is the assumption that GTP is preferred to GDP 
kgdp_onN = kon         # g'_d
kh = 1e-8*koff         # should be very slow: hydrolysis mainly occurs on ribosome

# derived parameter
ks = kh*kgdp_offN*kgtp_on/(K_D*kgdp_onN*kgtp_off)  # see paper eqn 7

# fixed parameters for ribosome cycle - correct amino acid
kc_on = kon    # k'_c
kc_off = koff  # k_c
lc_on = 1e-2*kon   # l'_c GTP-bound tRNA-complex much more likely to bind 
lc_off = koff  # l_c
mhc = 0.1*koff # m'
msc = (ks/kh)*mhc*lc_off*kc_on/(lc_on*kc_off)  # see paper eqn 9

# fixed parameters for ribosome cycle - wrong amino acid
kw_on = kc_on
kw_off = bias*koff  # D binds more weakly than C if bias > 1
lw_on = lc_on
lw_off = bias*lc_off
mhw = mhc
msw = (ks/kh)*mhw*lw_off*kw_on/(lw_on*kw_off)  # see paper eqn 9

#%% set up simulation 
# first find steady concentrations of C.GTP, C.GDP, W.GTP, and W.GDP. We assume unknown rxns hold
# [X], [GTP], [GDP], and [Pi] fixed and impose steady state to get [X.GTP], [X.GDP] where 
# X = either C (correct loaded tRNA) or W (wrong)
M = np.array([[-(kgtp_off+kh), conc_Pi*ks],[kh, -(conc_Pi*ks+kgdp_offN)]])
V = np.array([[kgtp_on*conc_GTP],[kgdp_onN*conc_GDP]])
conc_XGXP = -conc_C*np.dot(inv(M), V)
conc_CGTP = conc_XGXP[0][0]
conc_CGDP = conc_XGXP[1][0]
conc_WGTP = conc_XGXP[0][0]
conc_WGDP = conc_XGXP[1][0]
print("activation level = ",conc_CGTP/conc_CGDP)
#%% 
'''    states: 
            0=ribosome A site empty
            1=binds C.GTP (right)
            2=binds W.GTP (wrong)
            3=binds C.GDP
            4=binds W.GDP
'''
R = np.zeros((5,5)) # R[j,i]=rate to transition from state i to j, or 0 if j=i
R[0,0] = 0.
R[1,0] = conc_CGTP*kc_on
R[2,0] = conc_WGTP*kw_on
R[3,0] = conc_CGDP*lc_on
R[4,0] = conc_WGDP*lw_on
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
R[0,3] = lc_off + incorprate # rate to unbind or incorporate
R[1,3] = msc*conc_Pi
R[2,3] = 0.
R[3,3] = 0.
R[4,3] = 0.
#
R[0,4] = lw_off + incorprate # rate to unbind or incorporate
R[1,4] = 0.
R[2,4] = msw*conc_Pi
R[3,4] = 0.
R[4,4] = 0.
#%% run simulation
chains = []# placeholder for a list with simulated chains
ts = []    
fracs = [] # fraction of chain occupied by correct
states = []# what state after each transition
for i in range(numchains):
    print(i)
    chainM, tsM, statesM, wrongs = ribosim(R, incorprate, chainlength)
    chains = chains + [chainM]
    ts = ts + [tsM]
    states = states + [statesM]
    print(len(tsM)," transitions; ", wrongs," errors")
chains = np.array(chains)
ts = np.array(ts)
states = np.array(states)
np.savez('kproof', chains=chains, ts=ts, states=states)
