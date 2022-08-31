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

# rng = np.random.default_rng(69)
rng = np.random.default_rng()

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print('Start Time = ', start_time)

#%% Initialize

agent_list, target_list, env, kf, control = util.init_sim()
agent_list_original = agent_list.copy()

#%% Main Loop

simlen = 300 # seconds
maxiter = int(simlen/control.dt)
count = 0

for i in range(0, maxiter):
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
        # done = 0
        # for i2 in range(0, len(agent_list)):
        #     done += agent_list[i2][0].done
        # if done == len(agent_list):
        #     print("All agents have reached their target.")
        #     break
        
        msg = []
        for jj in range(0, len(agent_list)):
            ag2 = agent_list[jj][0]
            ag2.get_meas()
            # ag2.meas = util.miss_detect(ag2.rfs, ag2.meas)
            util.gen_clutter(ag2.rfs, env, ag2.meas)
            ag2.rfs.predict(dt=1.0)  # CPHD
            ag2.rfs.correct(meas=ag2.meas)
            ag2.rfs.prune()
            ag2.rfs.merge()
            ag2.rfs.cap()
            ag2.rfs.extract_states()
            ag2.tracked_obj = ag2.rfs.states
            ag2.make_broadcast()
         
        msg = []
        for qq in range(0, len(agent_list)):
            msg.append([])
        for xx in range(0, len(agent_list)):
            for zz in range(0, len(agent_list)):
                if xx == zz:
                    continue
                msg[xx].append(agent_list[zz][0].broadcast)
        for yy in range(0, len(agent_list)):
            msg2 = []
            ag3 = agent_list[yy][0]
            msg2.append(imp.DistributedGCI(msg[yy][0:2]))
            msg2.append(imp.DistributedGCI(msg[yy][2:4]))
            ag3.gm_fused = imp.DistributedGCI(msg2)
            if ag3.gm_fused:
                imp.CooperativeNavigation(ag3)
        
        count = count + 1
        if count % 10 == 0:
            print(count, 'sets of CPHD runs have been performed')

#%% Plots

util.plot_results(agent_list, target_list, env)

# agent_list[0][0].rfs.plot_states([0, 1], state_lbl='Agents',
#                                   lgnd_loc='lower left', state_color='g')

print('Start Time = ', start_time)
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print('End Time = ', end_time)
