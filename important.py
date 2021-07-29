# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 2021

@author: Vincent W. Hill

Important functions relating to paper "Multi-Sensor Fusion for Decentralized 
Cooperative Navigation Using Random Finite Sets" in IEEE Aerospace 2022

This differs from utilties in that these functions represent the core 
contributions of the paper.
"""

import numpy as np
import utilities as util

def dead_reckon(last_pos):
    """ Calculates dead reckoning position estimate.

        Args:
            last_pos: last timestep dead reckoning solution

        Returns:
            pos_est_dr: position estimate from dead reckoning solution
        """
    pos_est_dr = last_pos + 1
    return pos_est_dr

def identify_broadcast():
    """ Calculates likelihood that a received broadcast was sent by a tracked object.

        Args:
            rec_bc: received broadcast class object
            own_bc: own broadcast class object

        Returns:
            L_id: likelihood that broadcast was sent by tracked object
        """
    ans = 1 + 1
    return ans

def coop_nav():
    """ Calculates cooperative navigation position estimate.

        Args:
            dr_est: dead reckoning position estimate
            rec_bc: all received broadcast class objects
            own_bc: own broadcast class objects

        Returns:
            pos_est_cn: cooperative navigation position estimate
        """
    answer = 1 + 1
    return answer

def fuse():
    """ Fuses own and received CPHD solutions

        Args:
            rec_bc: received broadcast class object
            own_bc: own broadcast class object
            
        Returns:
            TBD
        """
    answer = 1 + 1
    return answer
