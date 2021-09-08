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
import scipy.stats as st
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
    
    xe = agent.pos_est[0].item()
    ye = agent.pos_est[1].item()
    u = agent.u
    v = agent.state[0].item()
    psi = agent.state[4].item()
    
    urand = rnd.uniform(-1*u/4, 1*u/4)
    vrand = rnd.uniform(-2, 5)
    # psirand = rnd.uniform(psi/4, 3*psi/4)
    
    pos = util.velprop(xe, ye, u + urand, v + vrand, psi, agent.dt)
    est = np.array([[pos[0]], [pos[1]]], dtype=float)
    
    agent.pos_est = est.copy()

def identify_broadcast(rec_bc, own_bc):
    """ Calculates likelihood that a received broadcast was sent by a tracked object.

        Args:
            rec_bc: list of received broadcast class objects
            own_bc: own broadcast class object

        Returns:
            L_id: likelihood that broadcast was sent by tracked object
        """
    nagent = len(rec_bc)
    rec_means = []
    own_means = []
    for i in range(0, nagent):
        rec_means.append(rec_bc[i].means[0].copy())
        rec_means.append(rec_bc[i].means[1].copy())
        own_means.append(own_bc.means[i].copy())
    L_id = 1
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
    if agent.gm_fused:
        if agent.gm_fused.means:
            agent.pos_est = np.array([[agent.gm_fused.means[0][0].item()],
                                      [agent.gm_fused.means[0][2].item()]])
        
def GeneralizedCovarianceIntersection(msg, inds):
    """ Fuses own and received CPHD solutions (must have same FoV)

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
    m = []
    af = []
    gf = []
    for i in range(0, len(inds)):
        m.append(inds[i][0])
        af.append(inds[i][1][0])
        gf.append(inds[i][1][1])
    w = 1/len(msg)  # metropolis weight
   
    weights = []
    means = []
    covs = []
    for i in range(0, len(m)):
        weights.append(msg[af[i]].weights[gf[i]])
        means.append(msg[af[i]].means[gf[i]].copy())
        covs.append(msg[af[i]].covariances[gf[i]].copy())
    
    w_temp, m_fuse, cov_fuse = fuse(w, weights, means, covs)
    
    w_sum = np.sum(w_temp)
    
    w_fuse = [x/w_sum for x in w_temp]
    
    gm_fused = GaussianMixture(means=m_fuse, 
                               covariances=cov_fuse, 
                               weights=w_fuse)
    
    return gm_fused

def fuse(omega, weight_list, mean_list, cov_list):
    """
    Fuse two Gaussian mixture components
    """
    lim = int(len(weight_list)/2)
    w_1 = weight_list[0:lim]
    w_2 = weight_list[lim:len(weight_list)]
    m_1 = mean_list[0:lim]
    m_2 = mean_list[lim:len(weight_list)]
    c_1 = cov_list[0:lim]
    c_2 = cov_list[lim:len(weight_list)]
    cov = []
    mean = []
    weight = []
    for i in range(0, lim):
        for j in range(0, lim):
            k0 = kernel(omega, c_1[i])
            k1 = kernel(1 - omega, c_2[j])
            cov_temp = la.inv(omega * la.inv(c_1[i]) + (1 - omega) * la.inv(c_2[j]))
            mean_temp = cov_temp @ (omega * la.inv(c_1[i]) @ m_1[i] + \
                                    (1 - omega) * la.inv(c_2[j]) @ m_2[j])
            weight_temp = w_1[i]**omega * w_2[j]**(1 - omega) * k0 * k1
            w_cov = c_1[i] / omega + c_2[j] / (1 - omega)
            ms = m_1[i] - m_2[j]
            weight_temp *= st.multivariate_normal.pdf(ms.flatten(),
                                                      mean=np.zeros(4),
                                                      cov=w_cov)
            if weight_temp > 1e-2:
                weight.append(weight_temp)
                cov.append(cov_temp.copy())
                mean.append(mean_temp.copy())
    return weight, mean, cov

def kernel(w, cov):
    num = la.det((2*np.pi/w)*cov)**0.5
    det = la.det(2*np.pi*cov)**(0.5*w)
    k = num/det
    return k

def DistributedGCI(msg):
    """
    Fuses CPHD solutions with different FoVs
    The implementation is based on
    @article{LI2021108210,
    title = {Distributed multi-view multi-target tracking based on CPHD filtering},
    journal = {Signal Processing},
    volume = {188},
    pages = {108210},
    year = {2021},
    issn = {0165-1684},
    doi = {https://doi.org/10.1016/j.sigpro.2021.108210},
    author = {Guchong Li and Giorgio Battistelli and Luigi Chisci and Wei Yi 
              and Lingjiang Kong},
    }
    """

    clus = cluster(msg, 1e2/2)
    do_gci = []
    for i in range(0, len(clus)):
        if clus[i]:
            do_gci.append((i, clus[i][0]))
    if do_gci:
        gm_fused = GeneralizedCovarianceIntersection(msg, do_gci)
    else:
        gm_fused = []
    return gm_fused
    
def mahalanobis(means, covs):
    temp = (means[0]-means[1]).T @ la.inv(covs[0] + covs[1]) @ (means[0]-means[1])
    dis = temp.item()
    return dis

def cluster(msg, rho):
    nagent = len(msg) # should be 2
    ngauss = len(msg[0].weights)
    idx = []
    gc = []
    clus = []
    for i in range(0, nagent):
        gc.append([])
    for i in range(0, nagent):
        for j in range(0, ngauss):
            idx.append((i, j))
            # gc[i].append((msg[i].means[j], msg[i].covariances[j]))
    clus = []
    for i in range(0, len(idx)):
        clus.append([])
    for num in range(0, len(idx)):
        ind = idx[num]
        i = ind[0]
        p = ind[1]
        idx2 = idx.copy()
        idx2.remove(ind)
        for k in range(0, len(idx2)):
            j = idx2[k][0]
            q = idx2[k][1]
            means = [msg[i].means[p]]
            covs = [msg[i].covariances[p]]
            means.append(msg[j].means[q])
            covs.append(msg[j].covariances[q])
            if mahalanobis(means, covs) < rho:
                clus[num].append((j,q))
    return clus