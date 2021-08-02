# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 2021

@author: Vincent W. Hill

Important functions relating to paper "Multi-Sensor Fusion for Decentralized 
Cooperative Navigation Using Random Finite Sets" in IEEE Aerospace 2022

This differs from utilties in that this script contains the new functions 
and core contributions of the paper.
"""

import numpy as np
import utilities as util
import gncpy.filters as filters
from scipy.linalg import block_diag

rng = np.random.default_rng(69)  # seeded sim
# rng = np.random.default_rng()     # unseeded sim

def get_meas(agent_list):
    meas = []
    for i in range(0, len(agent_list)):
        norm1 = agent_list[i][0].position[0].item()
        norm2 = agent_list[i][0].position[1].item()
        rg = np.sqrt(norm1**2 + norm2**2)
        
        bear = -np.arctan2(agent_list[i][0].position[1].item(), 
                           agent_list[i][0].position[0].item())
        meas.append(np.array([[bear], [rg]]))
    return meas

def dead_reckon(agent):
    """ Calculates dead reckoning position estimate.

        Args:
            agent: agent class object

        """
    rnd = np.random.default_rng()
    
    xe = agent.dr_pos_est[0].item()
    ye = agent.dr_pos_est[1].item()
    u = agent.u
    v = agent.state[0].item()
    psi = agent.state[4].item()
    
    urand = rnd.uniform(-3*u/4, 3*u/4)
    vrand = rnd.uniform(-5, 10)
    # psirand = rnd.uniform(psi/4, 3*psi/4)
    
    pos = util.velprop(xe, ye, u + urand, v + vrand, psi, agent.dt)
    est = np.array([[pos[0]], [pos[1]]], dtype=float)
    
    agent.dr_pos_est = est.copy()

def identify_broadcast(rec_bc, own_bc):
    """ Calculates likelihood that a received broadcast was sent by a tracked object.

        Args:
            rec_bc: received broadcast class object
            own_bc: own broadcast class object

        Returns:
            L_id: likelihood that broadcast was sent by tracked object
        """
    L_id = 1 + 1
    
    return L_id

def coop_nav(dr_est, rec_bc, own_bc):
    """ Calculates cooperative navigation position estimate.

        Args:
            dr_est: dead reckoning position estimate
            rec_bc: all received broadcast class objects
            own_bc: own broadcast class objects

        Returns:
            cn_est: cooperative navigation position estimate
        """
    cn_est = 1 + 1
    
    return cn_est

def fuse(rec_bc, own_bc):
    """ Fuses own and received CPHD solutions

        Args:
            rec_bc: received broadcast class object
            own_bc: own broadcast class object
            
        Returns:
            TBD
        """
    answer = 1 + 1
    
    return answer

def run_ukf(kf, kfsol, meas):
    prior = np.array([[kfsol[-1][0][0].item()], [2.], [kfsol[-1][0][1].item()], [2.]])
    cur_state = prior.copy()
    cur_input = np.array([[0.], [0.]])
    pred = kf.predict(cur_state=cur_state, cur_input=cur_input, dt=0.01)
    posterior = kf.correct(cur_state=pred, meas=meas)
    kfsol.append((np.array([[posterior[0][0].item()], [posterior[0][2].item()]]), 
                  posterior[1]))

def init_ukf(rfs, initprior):
    def meas_fnc(state, **kwargs):
        mag = state[0, 0]**2 + state[2, 0]**2
        sqrt_mag = np.sqrt(mag)
        mat = np.vstack((np.hstack((state[2, 0] / mag, 0,
                                    -state[0, 0] / mag, 0)),
                        np.hstack((state[0, 0] / sqrt_mag, 0,
                                   state[2, 0] / sqrt_mag, 0))))
        return mat
    
    def meas_mod(state, **kwargs):
        z1 = -np.arctan2(state[2, 0], state[0, 0])
        z2 = np.sqrt(state[0, 0]**2 + state[2, 0]**2)
        return np.array([[z1], [z2]])
    
    kf = filters.UnscentedKalmanFilter()
    kf.set_meas_mat(fnc=meas_fnc)
    kf.set_meas_model(meas_mod)
    kf.meas_noise = np.diag([(0.5 * np.pi / 180)**2, 5.**2])
    sig_w = 500.
    G = np.array([[0.01**2 / 2, 0],
                  [0.01,        0],
                  [0,         0.01**2 / 2],
                  [0,         0.01]])
    Q = block_diag(sig_w**2 * np.eye(2))
    kf.set_proc_noise(mat=G @ Q @ G.T)
    
    def dyn(x, **kwargs):
        dt = kwargs['dt']
        out = np.array([[x[0].item() + dt*x[1].item()],  # x
                        [x[1].item()],                   # xdot
                        [x[2].item() + dt*x[3].item()],  # y
                        [x[3].item()]])                  # ydotdot
        return out
    
    kf.dyn_fnc = dyn
    
    kf.cov = rfs.birth_terms[0].covariances[0]
    state0 = np.array([[initprior[-1][0][0].item()], [2.], 
                       [initprior[-1][0][1].item()], [2.]])
    alpha = 1e-3
    kappa = 0.
    kf.init_sigma_points(state0, alpha, kappa)
    
    return kf