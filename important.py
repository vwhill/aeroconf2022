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
import scipy.linalg as la
import utilities as util
from gasur.utilities.distributions import GaussianMixture
from copy import deepcopy

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

def CooperativeNavigation(agent):
    """ Calculates cooperative navigation position estimate.

        Args:
            dr_est: dead reckoning position estimate
            rec_bc: all received broadcast class objects
            own_bc: own broadcast class objects

        Returns:
            cn_est: cooperative navigation position estimate
        """
    dr_est = agent.dr_pos_est
    gm_fused = agent.gm_fused
    cn_est = 1 + 1
    
    return cn_est

def GeneralizedCovarianceIntersection(msg, agent):
    """ Fuses own and received CPHD solutions

        Args:
            msg: list of all messages containing 2DR&B measurements and
            Gaussian Mixture class
            
        Returns:
            gm_fused: fused Gaussian Mixture class
            
        The implementation is based on
        @ARTICLE{6472731,
          author={Battistelli, Giorgio and Chisci, Luigi and Fantacci, Claudio and 
                  Farina, Alfonso and Graziano, Antonio},
          journal={IEEE Journal of Selected Topics in Signal Processing}, 
          title={Consensus CPHD Filter for Distributed Multitarget Tracking}, 
          year={2013}, volume={7}, number={3}, pages={508-520}, 
          doi={10.1109/JSTSP.2013.2250911}}
        """
    w = 1/len(msg)  # metropolis weight
    meas_list = []
    weight_list = []
    mean_list = []
    cov_list = []
    for i in range(0, len(msg)):
        meas_list.append(msg[i][0])
        weight_list.append(msg[i][1].weights)
        mean_list.append(msg[i][1].means)
        cov_list.append(msg[i][1].covariances)
    
    mean_list_copy = deepcopy(mean_list)
    for i in range(0, len(mean_list_copy)):
        for j in range(0, len(mean_list_copy[i])):
            mean_list[i][j] = np.array([[agent.inv_meas[j][0].item()], 
                                        [mean_list_copy[i][j][1].item()], 
                                        [agent.inv_meas[j][1].item()], 
                                        [mean_list_copy[i][j][3].item()]])
    
    nagent = len(weight_list)
    ngauss = len(weight_list[0])
    weights = []
    means = []
    covs = []
    for i in range(0, nagent):
        weights.append(weight_list[i])
        means.append(mean_list[i])
        covs.append(cov_list[i])      
    
    w_temp, m_fuse, cov_fuse = fuse(w, weights, means, covs)
    
    w_sum = np.sum(w_temp)
    
    w_fuse = [x/w_sum for x in w_temp]
    
    gm_fused = GaussianMixture(means=m_fuse, 
                               covariances=cov_fuse, 
                               weights=w_fuse)
    
    return gm_fused

def fuse(omega, weight_list, mean_list, cov_list):
    cov = []
    mean = []
    weight = []
    for i in range(0, len(weight_list)):
        for j in range(0, len(weight_list)):
            k0 = kernel(omega, cov_list[0][i])
            k1 = kernel(1 - omega, cov_list[1][i])
            cov_temp = la.inv(omega * la.inv(cov_list[0][i]) + (1 - omega) * la.inv(cov_list[1][j]))
            mean_temp = cov_temp @ (omega * la.inv(cov_list[0][i]) @ mean_list[0][i] + \
                                    (1 - omega) * la.inv(cov_list[1][j]) @ mean_list[1][j])
            weight_temp = weight_list[0][i]**omega * weight_list[1][j]**(1 - omega) * k0 * k1
            w_cov = cov_list[0][i] / omega + cov_list[1][j] / (1 - omega)
            w_rand = rng.multivariate_normal(0*mean_temp.flatten(), w_cov)
            cov_temp2 = la.inv(la.inv(cov_temp) + la.inv(w_cov))
            weight.append(weight_temp)
            cov.append(cov_temp2.copy())
            mean.append(mean_temp.copy())
    return weight, mean, cov

def kernel(w, cov):
    num = la.det((2*np.pi/w)*cov)**0.5
    det = la.det(2*np.pi*cov)**(0.5*w)
    k = num/det
    return k