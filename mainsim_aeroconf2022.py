# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 2021

@author: Vincent W. Hill
Main simulation script for IEEE Aerospace 2022 paper "Multi-Sensor Fusion for
Decentralized Cooperative Navigation using Random Finite Sets"
"""

#%% Imports

import numpy as np
import utilities as util
import important as imp
import matplotlib.pyplot as plt
from datetime import datetime
import random

rng = np.random.default_rng(69)
random.seed()

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print('Start Time = ', start_time)

#%% Initialize

agent_list, target_list, env, kf, rfs, control = util.init_sim()

#%% Main Loop

simlen = 300 # seconds
maxiter = int(simlen/control.dt)
count = 0

for i in range(0, maxiter):
    meas = []
    for ii in range(0, len(agent_list)):
        a = agent_list[ii]
        a.state = a.state_mat @ a.state + a.input_mat @ a.desired_state
        a.save_state()
        a.prop_position()
        a.save_position()
        # a.get_meas()
        # meas.append(a.measurement)
        meas.append(a.position)
        a.check_waypoint()
    
    meas = util.miss_detect(rfs, meas)
    util.gen_clutter(rfs, env, meas)
    
    if i > 0: 
        if i % 100 == 0:
            rfs.predict()  # CPHD
            rfs.correct(meas=meas)
            rfs.prune()
            rfs.merge()
            rfs.cap()
            rfs.extract_states()

            count = count + 1
            if count % 10 == 0:
                print(count, 'CPHD runs have been performed')
                
            if count == 100:
                plt.figure()
                for i in range(0, len(agent_list)):
                    a = agent_list[i]
                    a.plot_position()
                for i in range(0, len(target_list)):
                    t = target_list[i]
                    t.plot_position()

                env.plot_obstacles()
                # plt.savefig('ground_truth_mid.pdf', format='pdf', transparent=True)
                
                rfsplot = rfs.plot_states([0, 1], meas_inds=[0, 1], state_lbl='Agents', lgnd_loc='lower left', state_color='g')
                # plt.savefig('rfsplot_mid.pdf', format='pdf',  transparent=True)

#%% Plots

rfs.plot_card_time_hist(lgnd_loc="lower left", sig_bnd = None) # sig_bnd = None
# plt.savefig('card_time_hist.pdf', format='pdf',  transparent=True)

plt.figure()
for i in range(0, len(agent_list)):
    a = agent_list[i]
    a.plot_position()

for i in range(0, len(target_list)):
    t = target_list[i]
    t.plot_position()

env.plot_obstacles()
# plt.savefig('ground_truth_end.pdf', format='pdf', transparent=True)

rfsplot = rfs.plot_states([0, 1], meas_inds=[0, 1], state_lbl='Agents', lgnd_loc='lower left', state_color='g')
# plt.savefig('rfsplot_end.pdf', format='pdf',  transparent=True)

print('Start Time = ', start_time)
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print('End Time = ', end_time)