# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 2021

@author: Vincent W. Hill

Important functions relating to paper "Multi-Sensor Fusion for Decentralized 
Cooperative Navigation Using Random Finite Sets" in IEEE Aerospace 2022

This differs from utilties in that these functions represent the new functions 
and core contributions of the paper.
"""

import numpy as np
import utilities as util

rng = np.random.default_rng(69)  # seeded sim
# rng = np.random.default_rng()     # unseeded sim

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
