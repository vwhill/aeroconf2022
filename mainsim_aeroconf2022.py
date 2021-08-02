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
from datetime import datetime

rng = np.random.default_rng(69)
# rng = np.random.default_rng()

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print('Start Time = ', start_time)

#%% Initialize

agent_list, target_list, env, kf, control = util.init_sim()

#%% Main Loop

simlen = 300 # seconds
maxiter = int(simlen/control.dt)
count = 0

for i in range(0, 500):
    for ii in range(0, len(agent_list)):
        a = agent_list[ii][0]
        num = agent_list[ii][1]
        a.prop_state()
        a.save_state()
        a.prop_position()
        imp.dead_reckon(a)
        a.save_position()
        a.get_object_positions(agent_list, num, env)
        a.check_waypoint()
      
    if i % 100 == 0:
        msg = []
        for jj in range(0, len(agent_list)):
        # for jj in range(0, 0):
            ag = agent_list[jj][0]
            ag.get_meas()
            # ag.meas = util.miss_detect(ag.rfs, ag.meas)
            # util.gen_clutter(ag.rfs, env, ag.meas)
            ag.rfs.predict(dt=1.0)  # CPHD
            ag.rfs.correct(meas=ag.meas)
            ag.rfs.prune()
            ag.rfs.merge()
            ag.rfs.cap()
            ag.rfs.extract_states()
            ag.make_broadcast()
            msg.append(agent_list[jj][0].broadcast)
        
        for yy in range(0, len(agent_list)):
            agent_list[yy][0].receive_broadcasts(msg)
        
        # fusion / CN

        count = count + 1
        if count % 10 == 0:
            print(count, 'sets of CPHD runs have been performed')

#%% Plots

util.plot_results(agent_list, target_list, env)

print('Start Time = ', start_time)
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print('End Time = ', end_time)
