# -*- coding: utf-8 -*-
"""
Python 3.7: ChenCode2B-bursting.py
Created  2019   (nelson@physics.upenn.edu)

Simulate the bursting model of RNA transcription
Supplement to Chen, Zuckerman, and Nelson "Stochastic Simulation to Visualize Gene 
Expression and Error Correction in Living Cells". Generates fig 3 of that article.
Inspired by a more complete simulation by Lok-Hang So dissertation http://hdl.handle.net/2142/24231
For more details see "Physical models of living systems" by Philip Nelson
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.close('all')

#%% set parameters and pre-allocate matrices
kclear = 0.5*np.log(2.)/50  # in 1/min (the 0.5 is obtained by eyeball fitting)
betastop = 1./6         # in 1/min (directly observed, not a a fit parameter)
betastart = 1./37       # in 1/min (directly observed, not a a fit parameter)
betasynth = 5*betastop  # in 1/min (Fano factor of 5 is directly observed, not a fit parameter)

N_runs = 1000   # how many runs to do
t_tot = 150    # in min
delta_t = 1    # how often to sample, min
N_steps = 130  # how many rxn steps

t_list = np.zeros((N_steps, N_runs))
N_list = np.zeros((N_steps, N_runs))


#%% Run simulation

for whichrun in range(N_runs):
    N_on = 0    #number of "on" copies, always 0 or 1
    num_rna = 0
    t = 0
    N_on_list = np.zeros((N_steps,1))
    rands = np.random.random(N_steps)
    for i_step in range(N_steps):
        #probabilities of switch state, synthesis of an mRNA, and
        # clearing of an mRNA
        p_switch = N_on*betastop + (1-N_on)*betastart
        p_synth = N_on * betasynth
        p_clear = num_rna * kclear

        p_norm = p_switch + p_synth + p_clear

        # random wait time for next reaction
        tau = -np.log(rands[i_step])/p_norm
        t += tau
        t_list[i_step, whichrun] = t    #end time of the current time interval

        # random reaction occurs
        rxn = np.random.choice(['switch','synth','clear'], \
                    p=[p_switch/p_norm, p_synth/p_norm, p_clear/p_norm])

        if rxn == 'switch':
            N_on = 1 - N_on
        elif rxn == 'synth':
            num_rna += 1
        elif rxn == 'clear':
            num_rna -= 1
        else:
            raise ValueError("invalid reaction type")

        N_list[i_step, whichrun] = num_rna  #population at the end of the time interval
        N_on_list[i_step] = N_on

        if t > t_tot:
            break
    if t < t_tot:
        print("ERROR- increase N_steps, time elapsed was",t)

#%% plot data

# show last run
#plt.figure(10); plt.plot(t_list[:i_step,whichrun]) # diagnostic
plt.figure(4)
plt.step(t_list[:(i_step-1), whichrun], N_on_list[:(i_step-1)],'r', where='post')
plt.step(t_list[:(i_step-1), whichrun], N_list[:(i_step-1), whichrun], 'b', where='post')
#%% re-express results of each simulation at evenly spaced times
commonlist = np.zeros((int(-1+t_tot/delta_t), N_runs))
for whichrun in range(N_runs):
    alpha = 0 # which step
    for q in np.arange(0, (-1+t_tot/delta_t), dtype='int'): # actual time in units of delta_t
        while t_list[alpha, whichrun] <= q*delta_t: # actual time in seconds
            alpha += 1
            if alpha>(N_steps): raise ValueError('increase N_steps')
        commonlist[q, whichrun] = N_list[alpha, whichrun]
#%% show the first twenty runs
plt.figure(1)
plt.plot(np.arange(0, (-1+t_tot/delta_t))*delta_t, commonlist[:,0:20])
plt.xlabel('time after induction  [min]'); plt.ylabel('number mRNA')

#%% compile list of how many have zero mRNA at each time
zerolist = np.zeros(int(-1+t_tot/delta_t))
for q in np.arange(0, (-1+t_tot/delta_t), dtype='int'):
    for j in range(N_runs):
        if commonlist[q,j] == 0:  zerolist[q] = zerolist[q] + 1
zerolist = zerolist/N_runs; # convert to estimated prob
#%% show <mRNA(t)>
plt.figure(2)
t_listb = np.arange(0, (-1+t_tot/delta_t))*delta_t # minutes
kmean = np.mean(commonlist,1);
plt.plot(t_listb, kmean, '.')
plt.title('Mean population versus time')
plt.xlabel('time after induction  [min]'); plt.ylabel(r'$\langle$number mRNA$\rangle$')
#%% show P(0)
plt.figure(3)
plt.plot(t_listb, np.log(zerolist), '.')
plt.title('Probability of zero copies versus time')
plt.xlabel('time after induction  [min]'); plt.ylabel(r'$\ln P_0(t)$')
#%% find fano factor
fano = np.var(commonlist[-1,:])/np.mean(commonlist[-1,:])
print('Fano factor = ', str(fano))
#%% finally add results from the naive birth-death model
fitA = 11; fitB = np.log(2.)/50 # a reasonable fit 
t_listc = np.linspace(0, t_tot, 100)
plt.figure(2)
plt.plot(t_listc, fitA*(1-np.exp(-t_listc*fitB)),'r')
plt.figure(3)
plt.plot(t_listc, -fitB*fitA*t_listc, 'r')
plt.ylim((-3.5,0.1))