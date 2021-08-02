# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 2021

@author: Vincent W. Hill

Utilities relating to paper "Multi-Sensor Fusion for Decentralized Cooperative 
Navigation Using Random Finite Sets" in IEEE Aerospace 2022

"""

import numpy as np
import math
from scipy import signal as sig
import scipy.linalg as la
import matplotlib.pyplot as plt
import gncpy.filters as filters
import gasur.swarm_estimator.tracker as track
from gasur.utilities.distributions import GaussianMixture
import gasur.utilities.graphs as graphs
from scipy.optimize import linear_sum_assignment as lsa
from scipy.linalg import block_diag

rng = np.random.default_rng(69) # seeded sim
# rng = np.random.default_rng() # unseeded sim

class Agent:
    def __init__(self):
        self.state = np.array([[]])
        self.position = np.array([[]])
        self.dr_pos_est = np.array([[]])
        self.cn_pos_est = np.array([[]])
        self.meas = np.array([[]])
        self.desired_state = np.array([[]])
        self.u = 5.
        self.dt = 0.01
        self.local_map = None

        self.state_mat = np.array([[]])
        self.input_mat = np.array([[]])
        self.meas_mat = np.array([[]])
        self.control_gain = np.array([[]])
        self.proc_noise = np.array([[]])
        self.meas_noise = np.array([[]])

        self.state_history = []
        self.position_history = []
        self.dr_pos_est_history = []
        self.cn_pos_est_history = []
        self.path = []
        self.waypoints = []
        self.current_waypoint = ([], 1)
        self.last_waypoint = ([], 0)
        self.dead = 0
        
        self.rfs = []
        self.obj_pos_true = []
        
    def check_waypoint(self):
        if self.current_waypoint[1] == len(self.waypoints):
            self.waypoints = list(reversed(self.waypoints))
        norm1 = self.dr_pos_est[0].item() - self.current_waypoint[0][0].item()
        norm2 = self.dr_pos_est[1].item() - self.current_waypoint[0][1].item()
        norm = np.sqrt(norm1**2 + norm2**2)
        if len(self.waypoints) <= self.current_waypoint[1]:
            self.current_waypoint = ((self.waypoints[1]), 1)
            self.last_waypoint = ((self.waypoints[0]), 0)
            return
        if norm < 30:
            self.last_waypoint = self.current_waypoint
            self.current_waypoint = (self.waypoints[self.current_waypoint[1]], 
                                     self.current_waypoint[1] + 1)
            self.desired_state[4] = guidance(self.dr_pos_est[0].item(),
                                             self.dr_pos_est[1].item(),
                                             self.current_waypoint[0][0].item(),
                                             self.current_waypoint[0][1].item())

            
    def prop_state(self, **kwargs):
        self.state = self.state_mat @ self.state + self.input_mat @ self.desired_state
        return self.state.copy()
    
    def save_state(self):
        self.state_history.append(self.state)
    
    def prop_position(self):
        xe = self.position[0].item()
        ye = self.position[1].item()
        u = self.u
        v = self.state[0].item()
        psi = self.state[4].item()
        pos = velprop(xe, ye, u, v, psi, self.dt)
        self.position = np.array([[pos[0]], [pos[1]]], dtype=float)
    
    def save_position(self):
        self.position_history.append(self.position)
        self.dr_pos_est_history.append(self.dr_pos_est)
        
    def plot_position(self):
        x = []
        y = []
        for i in range(0, len(self.position_history)):
            x.append(self.position_history[i][0])
            y.append(self.position_history[i][1])
        plt.plot(x, y , label='Agent trajectory', linewidth=2)
        plt.plot(self.position_history[0][0], self.position_history[0][1], 'ro', label='Initial Start')
        plt.plot(self.position_history[-1][0], self.position_history[-1][1], 'rx', label='Initial End')
        plt.xlabel('x-position')
        plt.ylabel('y-position')
    
    def plot_dr_pos_est(self):
        x = []
        y = []
        for i in range(0, len(self.dr_pos_est_history)):
            x.append(self.dr_pos_est_history[i][0])
            y.append(self.dr_pos_est_history[i][1])
        plt.plot(x, y , label='Dead Reckoning Navigation Solution', linewidth=2)
        plt.plot(self.dr_pos_est_history[0][0], self.dr_pos_est_history[0][1], 'ro', label='Start')
        plt.plot(self.dr_pos_est_history[-1][0], self.dr_pos_est_history[-1][1], 'rx', label='End')
        plt.xlabel('x-position')
        plt.ylabel('y-position')
    
    def get_meas(self, **kwargs):
        meas = []
        for i in range(0, len(self.obj_pos_true)):
            norm1 = self.obj_pos_true[i][0].item() - self.dr_pos_est[0].item()
            norm2 = self.obj_pos_true[i][1].item() - self.dr_pos_est[1].item()
            rg = np.sqrt(norm1**2 + norm2**2)
            # rg = rg + rng.normal()*rg
            bear = -math.atan2(self.obj_pos_true[i][1] - self.dr_pos_est[1],
                               self.obj_pos_true[i][0] - self.dr_pos_est[0])
            # bear = rng.normal()*bear
            meas.append(np.array([[bear], [rg]]))
        self.meas = meas
    
    def get_object_positions(self, agent_list, num, env):
        self.obj_pos_true = []
        for i in range(0, len(agent_list)):
            if i != num:
                self.obj_pos_true.append(agent_list[i][0].position)
        for j in range(0, len(env.obstacle_locations)):
            obs_pos = np.array([[env.obstacle_locations[j][0].item()], 
                                [env.obstacle_locations[j][1].item()]])
            self.obj_pos_true.append(obs_pos)

class Target:
    def __init__(self):
        self.position = []
    
    def plot_position(self):
        plt.plot(self.position[0], self.position[1], 'ro', label='Target Position')
        plt.xlabel('x-position')
        plt.ylabel('y-position')

class Environment:
    def __init__(self):
        self.max_x_inds = 0
        self.x_bnds = np.array([[]])
        self.x_side = 0
        self.max_y_inds = 0
        self.y_bnds = np.array([[]])
        self.y_side = 0
        self.map = np.array([[]])
        self.start = []
        self.end = []
        self.obstacle_list = []
        self.obstacle_locations = []

    def world_gen(self, obstacle_chance, x_side, y_side, **kwargs):
        self.x_bnds = (0, x_side)
        self.y_bnds = (0, y_side)
        self.pos_bnds = np.array([[self.x_bnds[0], self.x_bnds[1]],
                                  [self.y_bnds[0], self.y_bnds[1]]], dtype=float)
        self.x_side = x_side
        self.y_side = y_side
          
        self.map = rng.random((self.max_x_inds, self.max_y_inds))  # random obstacle locations
        for ii in range(0, np.size(self.map, axis=0)):
            for jj in range(0, np.size(self.map, axis=1)):
                if self.map[ii][jj] < obstacle_chance:
                    self.map[ii][jj] = 1
                    self.obstacle_list.append((ii, jj))
                    self.obstacle_locations.append(self.ind_to_pos(ii, jj))
                else:
                    self.map[ii][jj] = 0
        
        # self.map = np.zeros((self.max_x_inds, self.max_y_inds))  # set obstacle locations
        # self.map[0:int(1*self.max_y_inds/2), int(self.max_y_inds/3)] = 1
        # self.map[int(self.max_y_inds/5):int(4*self.max_y_inds/5), int(self.max_y_inds/3)] = 1
        # for ii in range(0, np.size(self.map, axis=0)):
        #     for jj in range(0, np.size(self.map, axis=1)):
        #         if self.map[ii][jj] == 1:
        #             self.obstacle_list.append((ii, jj))
        #             self.obstacle_locations.append(self.ind_to_pos(ii, jj))
        
        self.fix_obstacles()

    def fix_obstacles(self):
        for i in range(0, len(self.start)):
            self.map[self.start[i][0], self.start[i][1]] = 0
        for j in range(0, len(self.end)):
            self.map[self.end[j][0], self.end[j][1]] = 0

    def plot_obstacles(self):
        x_val = [x[0] for x in self.obstacle_locations]
        y_val = [x[1] for x in self.obstacle_locations]
        plt.scatter(x_val, y_val, s=30, label='Obstacles', color='black')

    def ind_to_pos(self, row_ind, col_ind):
        x_pos = col_ind/self.max_x_inds * self.x_side
        y_pos = row_ind/self.max_y_inds * self.y_side
        return np.array([[x_pos], [y_pos]], dtype=float)

    def pos_to_ind(self, x_pos, y_pos):
        row_ind = int(np.floor(self.max_y_inds * (y_pos / self.y_side)))
        col_ind = int(np.floor(self.max_x_inds * (x_pos / self.x_side)))
        return (row_ind, col_ind)

    def ind_has_obs(self, x_ind, y_ind):
        return self.map[y_ind, x_ind] > 0

    def pos_has_obs(self, x_pos, y_pos):
        row_ind, col_ind = self.pos_to_ind(x_pos, y_pos)
        return self.ind_has_obs(row_ind, col_ind)

    def set_targets(self, tar_pos_lst):
        for (x_p, y_p) in tar_pos_lst:
            row, col = self.pos_to_ind(x_p, y_p)
            self.map[row, col] = 0

    def display(self, **kwargs):
        f_hndl = kwargs.get('f_hndl', None)
        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        obs_row, obs_col = np.where(self.map > 0)
        x_pts = []
        y_pts = []
        for obj in zip(obs_row, obs_col):
            (x, y) = self.ind_to_pos(obj[0], obj[1])
            x_pts.append(x)
            y_pts.append(y)

        if len(x_pts) > 0:
            f_hndl.axes[0].scatter(x_pts, y_pts, color=(0, 0, 0),
                                   label='Obstacles')
        f_hndl.axes[0].set_xlim(self.x_bnds)
        f_hndl.axes[0].set_ylim(self.y_bnds)
        return f_hndl

class LateralFixedWing:
    def __init__(self):
        self.dt = 0.01
        self.statedim = 5
        self.indim = 2
                        #     v       p       r     phi    psi
        self.A = np.array([[-2.382,   0.,   -30.1,  65.49, 0.],  # v
                           [-0.702, -16.06,  0.872, 0.,    0.],  # p
                           [ 0.817, -16.65, -3.54,  0.,    0.],  # r
                           [ 0.,      1.,    0.,    0.,    0.],  # phi
                           [ 0.,      0.,    1.,    0.,    0.]]) # psi

                           # ail      rud
        self.B = np.array([[ 0.,    -7.41],  # v
                           [-36.3,  -688.],  # p
                           [-0.673, -68.0],  # r
                           [ 0.,      0.],   # phi
                           [ 0.,      0.]])  # psi

        self.C = np.eye(self.statedim)
        self.D = np.eye(self.statedim, self.indim)
        [self.F, self.G] = discretize(self.dt, self.A, self.B, self.C, self.D)

class DoubleIntegrator:
    def __init__(self):
        self.dt = 0.01
        self.A = np.array([[0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0]])

        self.B = np.array([[0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 0.0],
                           [0.0, 1.0]])

        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

        [self.F, self.G] = discretize(self.dt, self.A, self.B, self.C, np.zeros((2, 2)))

class LQR:
    def __init__(self, F, G, statedim, indim):
        self.dt = 0.01
        C = np.identity(statedim)
        D = np.zeros((statedim, statedim))
        Q = np.identity(statedim)
        R = np.identity(indim)
        P = la.solve_discrete_are(F, G, Q, R, e=None, s=None, balanced=True)
        K = la.inv(G.T@P@G+R)@(G.T@P@F)

        sysac = sig.StateSpace(F-G@K, G@K, C, D, dt=self.dt)

        self.F = sysac.A
        self.G = sysac.B

def discretize(dt, A, B, C, D):
    sys = sig.StateSpace(A, B, C, D)
    sys = sys.to_discrete(dt)
    F = sys.A
    G = sys.B
    return F, G

def guidance(xa, ya, xt, yt):
    Hdes = float(-math.atan2((yt-ya), (xt-xa)))
    if np.abs(yt-ya) < 10:
        if xt-xa > 0:
            Hdes = 0
        if xt-xa < 0:
            Hdes = 3.14159
    return Hdes

def velprop(xe, ye, u, v, psi, dt):
    uv = np.array(([float(u)], [float(v)]))
    dcm = np.array([[float(np.cos(psi)), float(np.sin(psi)),],
                    [float(-np.sin(psi)), float(np.cos(psi))]])
    xy = dcm@uv
    xp = float(xe+dt*xy[0])
    yp = float(ye+dt*xy[1])
    return xp, yp

def agent_death(agent_list, rfs_agent):
    for i in range(0, len(agent_list)):
        agent = agent_list[i]
        if rng.random() > rfs_agent.prob_survive:
            agent.dead = 1
            break
    agent_list_survive = []
    for i in range(0, len(agent_list)):
        agent = agent_list[i]
        if agent.dead == 0:
            agent_list_survive.append(agent)
    return agent_list_survive

def init_astar(agent_list, env):
    all_paths = []
    cost_mat = np.zeros((len(env.start), len(env.end)))
    
    for i in range(0, len(env.start)):
        temp = []
        for ii in range(0, len(env.end)):
            path_tup = graphs.a_star_search(env.map, env.start[i], env.end[ii])
            temp.append(path_tup[0])
            cost_mat[i, ii] = path_tup[1]
        all_paths.append(temp)
    
    [row_ind, col_ind] = lsa(cost_mat)
    
    for i in range(0, len(agent_list)):
        ag = agent_list[i][0]
        ag.path = all_paths[i][col_ind[i]]
        wpts = []
        for ii in range(0, len(ag.path)):
            wpts.append(env.ind_to_pos(ag.path[ii][0], ag.path[ii][1]))
        ag.waypoints = wpts
        ag.current_waypoint = (wpts[1], 1)
        ag.last_waypoint = (wpts[0], 0)
        Hdes = guidance(ag.position[0], ag.position[1],
                    ag.current_waypoint[0][0], ag.current_waypoint[0][1])
        ag.desired_state[4] = Hdes
    return agent_list

def swarm_guidance_update(agent_list, target_list, rfs_agent, rfs_target, env):
    for i in range(0, len(agent_list)):
        curr = agent_list[i][0]
        env.start[i] = env.pos_to_ind(curr.position[0].item(), 
                                      curr.position[1].item())
    for i in range(0, len(target_list)):
        curr = target_list[i]
        env.end[i] = env.pos_to_ind(curr.position[0].item(), 
                                    curr.position[1].item())
        
    env.fix_obstacles()
    agent_list = init_astar(agent_list, env)
    
    return agent_list

def plot_results(agent_list, target_list, rfs, env, kfsol, ukfsol):
    # rfs.plot_card_time_hist(lgnd_loc="lower left", sig_bnd = None) # sig_bnd = None
    # plt.savefig('card_time_hist.pdf', format='pdf',  transparent=True)

    plt.figure()
    for i in range(0, len(agent_list)):
        a = agent_list[i][0]
        a.plot_position()
    # for i in range(0, len(target_list)):
    #     t = target_list[i]
    #     t.plot_position()

    env.plot_obstacles()
    plt.title('True Position')
    # plt.savefig('ground_truth.pdf', format='pdf', transparent=True)
    
    # plt.figure()
    # x = []
    # y = []
    # for i in range(0, len(kfsol)):
    #     x.append(kfsol[i][0][0])
    #     y.append(kfsol[i][0][1])
    # plt.plot(x, y , label='Agent Trajectory', linewidth=2)
    # plt.plot(kfsol[0][0][0], kfsol[0][0][1], 'ro', label='Initial Start')
    # plt.plot(kfsol[-1][0][0], kfsol[-1][0][1], 'rx', label='Initial End')
    # plt.xlabel('x-position')
    # plt.ylabel('y-position')
    # plt.title('EKF Position Estimate')
    
    # x = []
    # y = []
    # for i in range(0, len(ukfsol)):
    #     x.append(ukfsol[i][0][0])
    #     y.append(ukfsol[i][0][1])
    # plt.plot(x, y , label='Agent Trajectory', linewidth=2)
    # plt.plot(ukfsol[0][0][0], ukfsol[0][0][1], 'ro', label='Initial Start')
    # plt.plot(ukfsol[-1][0][0], ukfsol[-1][0][1], 'rx', label='Initial End')
    # plt.xlabel('x-position')
    # plt.ylabel('y-position')
    # plt.title('UKF Position Estimate')
    
    rfs.plot_states([0, 2], state_lbl='Object States', lgnd_loc='lower left',
                    state_color='g', sig_bnd=None)
    # plt.savefig('rfsplot.pdf', format='pdf',  transparent=True)
    
    # plt.figure()
    # for i in range(0, len(agent_list)):
    #     a = agent_list[i][0]
    #     a.plot_dr_pos_est()
        
    # # for i in range(0, len(target_list)):
    # #     t = target_list[i]
    # #     t.plot_position()

    # env.plot_obstacles()
    # plt.title('DR Position Estimate')
    # plt.savefig('dead_reckoning_solution.pdf', format='pdf', transparent=True)
    
    # rfs.plot_card_dist()

def gen_clutter(rfs, env, meas):  # need to update for 2DR&B
    y = []    
    num_clutt = rng.poisson(rfs.clutter_rate)
    for ff in range(num_clutt):
        m = (env.pos_bnds[:, [1]] - env.pos_bnds[:, [0]]) * rng.random((2, 1))
        y.append(m)
    
    for i in range(0, len(y)):
        norm1 = y[i][0].item() - (env.pos_bnds[0][1].item())/2
        norm2 = y[i][1].item() - (env.pos_bnds[1][1].item())/2
        rg = np.sqrt(norm1**2 + norm2**2)
        bear = math.atan2(y[i][1] - env.pos_bnds[0][1].item(),
                          y[i][0] - env.pos_bnds[1][1].item())
        meas.append(np.array([[bear], [rg]]))

def miss_detect(rfs, meas):
    y = []
    for i in range(0, len(meas)):
        if rng.uniform() > rfs.prob_miss_detection:
            y.append(meas[i])
    return y

def run_ekf(kf, kfsol, meas):
    prior = np.array([[kfsol[-1][0][0].item()], [2.], [kfsol[-1][0][1].item()], [2.]])
    cur_state = prior.copy()
    cur_input = np.array([[0.], [0.]])
    pred = kf.predict(cur_state=cur_state, cur_input=cur_input, dt=0.01)
    posterior = kf.correct(cur_state=pred, meas=meas)
    kfsol.append((np.array([[posterior[0][0].item()], [posterior[0][2].item()]]), 
                  posterior[1]))

def init_cphd(agent_list, kf, env):
    gm = []
    cov = 10.0**2*np.eye(4)
    cov[1, 1] = 5.0**2
    cov[3, 3] = 5.0**2
    for i in range(0, len(agent_list)):
        vec = np.array([[agent_list[i][0].position[0].item()], [2.], 
                        [agent_list[i][0].position[1].item()], [2.]])
        gm.append(GaussianMixture(means=[vec], 
                                  covariances=[cov.copy()], weights=[1/len(agent_list)]))

    rfs = track.CardinalizedPHD()
    # rfs = track.ProbabilityHypothesisDensity()
    rfs.save_covs = True
    rfs.max_expected_card = len(env.start) + 1
    rfs.prob_detection = 0.999
    rfs.prob_survive = 0.99
    rfs.merge_threshold = 4
    rfs.req_births = len(env.start) + 1
    rfs.birth_terms = gm
    rfs.inv_chi2_gate = 32.2361913029694
    rfs.gating_on = False
    rfs.clutter_rate = 1.0
    rfs.clutter_den = 1 / np.prod(env.pos_bnds[:, [1]] - env.pos_bnds[:, [0]])
    rfs.filter = kf

    return rfs

def init_sim():
    env = Environment()
    obs_chance = 0.001
    x_side = 2000
    y_side = 2000
    env.max_x_inds = 26
    env.max_y_inds = 26
    
    env.world_gen(obs_chance, x_side, y_side)
    
    env.start = [(1, 0),
                 (12, 0),
                 (16, 0),
                 (20, 0)]
    # env.start = [env.start[3]]
    
    env.end = [(1, env.max_x_inds-1),
               (8, env.max_x_inds-1),
               (17, env.max_x_inds-1),
               (25, env.max_x_inds-1)]   
    # env.end = [env.end[0]]
    
    env.fix_obstacles()
    
    fw  = LateralFixedWing()
    di = DoubleIntegrator()
    control = LQR(fw.F, fw.G, fw.statedim, fw.indim)
    
    proc_noise = 100.0*np.ones((4, 1))
    proc_noise[2, 0] = 20.0
    proc_noise[3, 0] = 20.0
    meas_noise = 10.0*np.ones((2, 1))
    
    Q = np.zeros((4, 4))
    R = np.zeros((2, 2))
    
    for i in range(0, len(proc_noise)):
        Q[i, i] = proc_noise[i]
    for j in range(0, len(meas_noise)):
        R[j, j] = meas_noise[j]
    
    agent_list = []
    target_list = []
    
    for i in range(0, len(env.start)):
        a = Agent()
        a.state = np.zeros((fw.statedim, 1))
        a.save_state()
        a.position = env.ind_to_pos(env.start[i][0], env.start[i][1])
        a.dr_pos_est = a.position.copy()
        a.save_position()
        a.desired_state = np.zeros((fw.statedim, 1))
        a.desired_state[4] = np.deg2rad(0.)
        a.state_mat = control.F
        a.input_mat = control.G
        a.meas_mat = np.zeros((2, fw.statedim))
        a.proc_noise = Q
        a.meas_noise = R
        a.dt = fw.dt
        agent_list.append((a, i))
    
    for j in range(0, len(env.end)):
        t = Target()
        t.position = env.ind_to_pos(env.end[j][0], env.end[j][1])
        target_list.append(t)
    
    agent_list = init_astar(agent_list, env)
    
    for i in range(0, len(agent_list)):
        a = agent_list[i][0]
        a.check_waypoint()
    
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
    
    # kf = filters.UnscentedKalmanFilter()
    kf = filters.ExtendedKalmanFilter()
    kf.set_meas_mat(fnc=meas_fnc)
    kf.set_meas_model(meas_mod)
    kf.meas_noise = np.diag([(10. * np.pi / 180)**2, 10.0**2])
    sig_w = 10.
    sig_u = np.pi / 180
    G = np.array([[1.**2 / 2, 0],
                  [1.,        0],
                  [0,         1.**2 / 2],
                  [0,         1.]])
    Q = block_diag(sig_w**2 * np.eye(2))
    kf.set_proc_noise(mat=G @ Q @ G.T)
    
    # def dyn(x, **kwargs):
    #     dt = kwargs['dt']
    #     out = np.array([[x[0].item() + dt*x[1].item()],  # x
    #                     [x[1].item()],                   # xdot
    #                     [x[2].item() + dt*x[3].item()],  # y
    #                     [x[3].item()]])                  # ydotdot
    #     return out
    
    # kf.dyn_fnc = dyn
    
    # kf.cov = 10.0**2*np.eye(4)
    # kf.cov[1, 1] = 5.0**2
    # kf.cov[3, 3] = 5.0**2
    # state0 = np.array([[agent_list[0][0].position[0].item()], [2.], 
    #                     [agent_list[0][0].position[1].item()], [2.]])
    # alpha = 1.0
    # kappa = 2.0
    # kf.init_sigma_points(state0, alpha, kappa)
    
        # returns x_dot
    def f0(x, u, **kwargs):
        return x[1]
    
    
    # returns x_dot_dot
    def f1(x, u, **kwargs):
        return 0.
    
    
    # returns y_dot
    def f2(x, u, **kwargs):
        return x[3]
    
    
    # returns y_dot_dot
    def f3(x, u, **kwargs):
        return 0.
    
    kf.dyn_fncs = [f0, f1, f2, f3]
    
    rfs = init_cphd(agent_list, kf, env)
    
    return agent_list, target_list, env, kf, rfs, control


