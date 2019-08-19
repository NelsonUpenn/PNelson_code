# -*- coding: utf-8 -*-
"""
Python: 3.8   ChenCode2-transcrip.py
Created 2019  (nelson@physics.upenn.edu)
Supplement to Chen, Zuckerman, and Nelson "Stochastic Simulation to Visualize Gene 
Expression and Error Correction in Living Cells"

Description: create and display stochastic simulation of birth-death process
 Generate Fig 1 of "Stochastic Simulation to Visualize Gene Expression and Error Correction in Living Cells"
 For more details see "Physical models of living systems" by Philip Nelson
 
"""
import numpy as np, matplotlib.pyplot as plt

# Main code begins at MAIN below. 
# The following function is the Gillespie simulation engine.

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
        if t>treport:                     # issue progress reports
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

#%% MAIN
plt.close('all')
myfigsizea = (3, 2.5); myfigsizeb = (2.5,2)
#example with large populations (figure panels a,b)
ks = np.array([12, .015]) # rate constants in 1/min
runtime= 1600
Nruns = 100
lini = 3200
coarse_histo = 16  #lump histogram bins
# example with small population (figure panels c,d), same relaxation time constant
'''ks = np.array([.15, .015]) # rate constants in 1/min
runtime= 1600
Nruns=150 
lini = 40
coarse_histo = 1'''
#%% show two examples of time series
plt.figure(frameon=False, figsize=myfigsizea)
(ts, ls) = transcrip2rxn(lini, runtime, ks)
plt.step(ts, ls) 
ax = plt.gca()
ax.set_xlabel('time, min'); ax.set_ylabel('population')
(ts, ls) = transcrip2rxn(lini, runtime, ks)
plt.step(ts, ls, 'r'); 
plt.ylim(0,lini*1.01)
#%% compare deterministic model
tmax = np.max(ts);
tvals  = np.arange(0,tmax,(tmax/100))
plt.plot(tvals, (ks[0]/ks[1])*(1-np.exp(-ks[1]*tvals)) + lini*np.exp(-ks[1]*tvals),'k')
ax.set_title('duration= '+ str(runtime)+ '; ks= '+ str(ks[0])+ ', ' +str(ks[1]))

#%% compare average over many trials
final=np.zeros(Nruns, dtype='int'); finalt=np.zeros(Nruns)
allhistory=np.zeros((Nruns,runtime))
for whichrun in range(Nruns):
    (ts, ls) = transcrip2rxn(lini, runtime, ks)
    final[whichrun] = ls[-1]
    finalt[whichrun] = ts[-1]
    beancount = 0; # tells us which event we have passed by now
    for realtime in range(runtime):
        while realtime > ts[beancount]:
            beancount = beancount+1
            if beancount > len(ls): break
#realtime has passed one or more events, so update:
        if beancount>0: allhistory[whichrun,realtime]=ls[beancount-1]
#%%
print('estimated pop expectation = '+str(np.mean(final))+ '; var = ' +str(np.var(final)))
print('estimated elapsed time expectation = '+ str(np.mean(finalt)) +'; var = '+ str(np.var(finalt)))
#%% ensemble averaged time course
tvals = np.arange(runtime)
tvals = tvals[1:]; allhistory = allhistory[:, 1:] # drop first uninteresting value
plt.plot(tvals, np.mean(allhistory,0),'g')
#%% histogram final states
plt.figure(frameon=False, figsize=myfigsizeb)
xmax = int(np.max(final)) + 2
binlist = np.arange(-.1, xmax, coarse_histo) # first bin straddles the range 0,...,(coarse_histo-1)
counts, _ = np.histogram(final, binlist)
plt.bar(binlist[:-1]-0.4, counts/(coarse_histo*Nruns), width=coarse_histo*.9, align='edge')
#%%% compare Poisson
mu = ks[0]/ks[1]
logpdist = np.zeros(xmax)
logpdist[0] = -mu
for i in range(1,xmax):
    logpdist[i] = logpdist[i-1] + np.log(mu/i)
print('check: ', mu,'=',np.sum(np.exp(logpdist)*np.arange(0,xmax)))
plt.plot(np.arange(0,xmax), np.exp(logpdist),'r-')
a2 = plt.gca()
a2.set_xlabel('population'); a2.set_ylabel('estimated probability')
