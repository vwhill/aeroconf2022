# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 2021

@author: Vincent W. Hill
Main simulation script for IEEE Aerospace 2022 paper "Multi-Sensor Fusion for
Decentralized Cooperative Navigation using Random Finite Sets"
ag"""

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

for i in range(0, 505):
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
    
    if i % 101 == 0:
        msg = []
        for jj in range(0, len(agent_list)):
        # for jj in range(0, 0):
            ag2 = agent_list[jj][0]
            ag2.get_meas()
            # ag.meas = util.miss_detect(ag.rfs, ag.meas)
            # util.gen_clutter(ag.rfs, env, ag.meas)
            ag2.rfs.predict(dt=1.0)  # CPHD
            ag2.rfs.correct(meas=ag2.meas)
            ag2.rfs.prune()
            ag2.rfs.merge()
            ag2.rfs.cap()
            ag2.rfs.extract_states()
            ag2.tracked_obj = ag2.rfs.states
            ag2.make_broadcast()
            msg.append(ag2.broadcast)
        
        msg = []
        for xx in range(0, len(agent_list)):
            msg.append(agent_list[xx][0].broadcast)
        # for yy in range(0, len(agent_list)):
        for yy in range(1, 2):
            ag3 = agent_list[yy][0]
            ag3.receive_broadcasts(msg)
            # ag3.gm_fused = imp.GeneralizedCovarianceIntersection(msg, ag3)
            # ag3.cn_pos_est = imp.CooperativeNavigation(ag3)
            # ag3.cn_pos_est_hist.append(ag3.cn_pos_est)

        count = count + 1
        if count % 10 == 0:
            print(count, 'sets of CPHD runs have been performed')

#%% Plots

util.plot_results(agent_list, target_list, env)

print('Start Time = ', start_time)
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print('End Time = ', end_time)
