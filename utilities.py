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
import numpy.random as rnd
import scipy.linalg as la
import matplotlib.pyplot as plt
import gncpy.filters as filters
import gasur.swarm_estimator.tracker as track
from gasur.utilities.distributions import GaussianMixture
import gasur.utilities.graphs as graphs
from scipy.optimize import linear_sum_assignment as lsa

rng = np.random.default_rng(69)

class Agent:
    def __init__(self):
        self.state = np.array([[]])
        self.position = np.array([[]])
        self.measurement = np.array([[]])
        self.desired_state = np.array([[]])
        self.u = 10
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
        self.path = []
        self.waypoints = []
        self.current_waypoint = ([], 1)
        self.last_waypoint = ([], 0)
        self.dead = 0
        
    def check_waypoint(self):
        if self.current_waypoint[1] == len(self.waypoints):
            self.waypoints = list(reversed(self.waypoints))
        norm1 = self.position[0] - self.current_waypoint[0][0]
        norm2 = self.position[1] - self.current_waypoint[0][1]
        norm = np.sqrt(norm1**2 + norm2**2)
        if len(self.waypoints) <= self.current_waypoint[1]:
            self.current_waypoint = ((self.waypoints[1]), 1)
            self.last_waypoint = ((self.waypoints[0]), 0)
            return
        if norm < 30:
            self.last_waypoint = self.current_waypoint
            self.current_waypoint = (self.waypoints[self.current_waypoint[1]], 
                                     self.current_waypoint[1] + 1)
            self.desired_state[4] = guidance(self.last_waypoint[0][0],
                                             self.last_waypoint[0][1],
                                             self.current_waypoint[0][0],
                                             self.current_waypoint[0][1])

            
    def prop_state(self, **kwargs):
        rng = kwargs.get('rng', rnd.default_rng())
        n = rng.multivariate_normal(mean=np.zeros(self.state.size), cov=self.proc_noise)
        n = n.reshape(self.state.shape)
        self.state = self.state_mat @ self.state + self.input_mat @ self.desired_state + n
        return self.state.copy()

    def get_meas(self, **kwargs):  # Need to update for 2DR&B
        rng = kwargs.get('rng', rnd.default_rng())
        n = rng.multivariate_normal(mean=np.zeros(self.meas_noise.shape[0]), cov=self.meas_noise)
        n = n.reshape((self.meas_mat.shape[0], 1))
        self.measurement = self.meas_mat @ self.state + n
        return self.measurement.copy()
    
    def save_state(self):
        self.state_history.append(self.state)
    
    def save_position(self):
        self.position_history.append(self.position)
        
    def plot_position(self):
        x = []
        y = []
        for i in range(0, len(self.position_history)):
            x.append(self.position_history[i][0])
            y.append(self.position_history[i][1])
        plt.plot(x, y , label='Aircraft path', linewidth=2)
        plt.plot(self.position_history[0][0], self.position_history[0][1], 'ro', label='Initial Start')
        plt.plot(self.position_history[-1][0], self.position_history[-1][1], 'rx', label='Initial End')
        plt.xlabel('x-position')
        plt.ylabel('y-position')

    def calc_control(self, waypoint):
        ctrl = self.control_gain @ (waypoint - self.state)
        lim = 5
        for ii in range(0, ctrl.size):
            ctrl[ii, 0] = min(ctrl[ii, 0], lim)
            ctrl[ii, 0] = max(ctrl[ii, 0], -lim)
        return ctrl
    
    def prop_position(self):
        xe = self.position[0]
        ye = self.position[1]
        u = self.u
        v = self.state[0]
        psi = self.state[4]
        pos = velprop(xe, ye, u, v, psi, self.dt)
        self.position = np.array([[pos[0]], [pos[1]]])

class Target:
    def __init__(self):
        self.position = np.array([[]])
    
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
                                  [self.y_bnds[0], self.y_bnds[1]]])
        self.x_side = x_side
        self.y_side = y_side
        
        rng = kwargs.get('rng', rnd.default_rng())    
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
        return (x_pos, y_pos)

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
                           [0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]])

        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0]])

        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

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
    # d2d = float(np.sqrt((xa-xt)**2+(ya-yt)**2))
    Hdes = float(-math.atan2((yt-ya), (xt-xa)))
    if np.abs(yt-ya) < 10:
        if xt-xa > 0:
            Hdes = 0
        if xt-xa < 0:
            Hdes = 3.14159
    # return d2d, Hdes
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
        ag = agent_list[i]
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
        curr = agent_list[i]
        env.start[i] = env.pos_to_ind(curr.position[0].item(), 
                                      curr.position[1].item())
    for i in range(0, len(target_list)):
        curr = target_list[i]
        env.end[i] = env.pos_to_ind(curr.position[0].item(), 
                                    curr.position[1].item())
        
    env.fix_obstacles()
    agent_list = init_astar(agent_list, env)
    
    return agent_list

def init_cphd(agent_list, kf, env):
    birth = []
    gm = []
    cov = np.zeros((2, 2))
    cov = 100.0*np.eye(4)
    cov[2, 2] = 20.0
    cov[3, 3] = 20.0
    for i in range(0, len(agent_list)):
        vec = np.array([[agent_list[i].position[0]], 
                           [agent_list[i].position[1]], [0], [0]])
        gm.append(GaussianMixture(means=[vec], 
                                  covariances=[cov.copy()], weights=[1]))
        birth.append((gm[i], 0.03))

    rfs = track.CardinalizedPHD()
    rfs.max_expected_card = len(env.start)
    rfs.prob_detection = 0.999
    rfs.prob_survive = 0.99
    rfs.merge_threshold = 4
    rfs.req_births = len(env.start)
    rfs.birth_terms = gm
    rfs.inv_chi2_gate = 32.2361913029694
    rfs.gating_on = False
    rfs.clutter_rate = 1
    rfs.clutter_den = 1 / np.prod(env.pos_bnds[:, [1]] - env.pos_bnds[:, [0]])
    rfs.filter = kf
    
    return rfs

def gen_clutter(rfs, env, meas):
    num_clutt = rng.poisson(rfs.clutter_rate)
    for ff in range(num_clutt):
        m = (env.pos_bnds[:, [1]] - env.pos_bnds[:, [0]]) * rng.random((2, 1))
        meas.append(m)
    return meas

def miss_detect(rfs, meas):
    y = []
    for i in range(0, len(meas)):
        if rng.uniform() > rfs.prob_miss_detection:
            y.append(meas[i])
    return y

def init_sim():
    env = Environment()
    obs_chance = 0.001
    x_side = 2000
    y_side = 2000
    env.max_x_inds = 26
    env.max_y_inds = 26
    
    env.world_gen(obs_chance, x_side, y_side)
    
    env.start = [(8, 0),
                 (12, 0),
                 (16, 0),
                 (20, 0)]
    
    env.end = [(1, env.max_x_inds-1),
               (7, env.max_x_inds-1),
               (12, env.max_x_inds-1),
               (17, env.max_x_inds-1)]    
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
    
    kf = filters.KalmanFilter()
    kf.set_proc_noise(mat=Q)
    kf.meas_noise = R
    kf.set_state_mat(mat=di.F)
    kf.set_input_mat(mat=di.G)
    kf.set_meas_mat(mat=di.C)
    
    agent_list = []
    target_list = []
    
    for i in range(0, len(env.start)):
        a = Agent()
        a.state = np.zeros((fw.statedim, 1))
        a.save_state()
        a.position = env.ind_to_pos(env.start[i][0], env.start[i][1])
        a.save_position()
        a.desired_state = np.zeros((fw.statedim, 1))
        a.desired_state[4] = np.deg2rad(0)
        a.state_mat = control.F
        a.input_mat = control.G
        a.meas_mat = np.zeros((2, fw.statedim))
        a.proc_noise = Q
        a.meas_noise = R
        a.dt = fw.dt
        agent_list.append(a)
    
    for j in range(0, len(env.end)):
        t = Target()
        t.position = env.ind_to_pos(env.end[j][0], env.end[j][1])
        target_list.append(t)
    
    agent_list = init_astar(agent_list, env)
    
    for i in range(0, len(agent_list)):
        a = agent_list[i]
        a.check_waypoint()
    
    rfs = init_cphd(agent_list, kf, env)
    
    return agent_list, target_list, env, kf, rfs, control
