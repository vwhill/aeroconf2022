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
from copy import deepcopy
from scipy.optimize import fsolve
from rl.random import OrnsteinUhlenbeckProcess

# rng = np.random.default_rng(69) # seeded sim
rng = np.random.default_rng() # unseeded sim

class Agent:
    def __init__(self):
        self.state = np.array([[]])
        self.position = np.array([[]])
        self.pos_est = np.array([[]])
        self.meas = np.array([[]])
        self.desired_state = np.array([[]])
        self.u = 10.
        self.dt = 0.01
        self.done = 0

        self.state_mat = np.array([[]])
        self.input_mat = np.array([[]])
        self.meas_mat = np.array([[]])
        self.control_gain = np.array([[]])
        self.proc_noise = np.array([[]])
        self.meas_noise = np.array([[]])

        self.state_history = []
        self.position_history = []
        self.pos_est_hist = []
        self.waypoints = []
        self.current_waypoint = ([], 1)
        self.last_waypoint = ([], 0)
        self.dead = 0
        
        self.rfs = track.CardinalizedPHD()
        self.obj_pos_true = []
        self.tracked_obj = []
        self.broadcast = []
        self.broadcast_hist = []
        self.messages = []
        self.message_hist = []
        self.gm_fused = []
        self.inv_meas = []
        
        self.rp_u = OrnsteinUhlenbeckProcess(theta=0.01, mu=0., sigma=0.4, size=1)
        self.rp_v = OrnsteinUhlenbeckProcess(theta=1, mu=0., sigma=10., size=1)
        
    def check_waypoint(self):
        if self.done == 1:
            self.desired_state[4] = CoordinatedTurn(self)
            return
        if self.current_waypoint[1] == len(self.waypoints):
            # self.waypoints = list(reversed(self.waypoints))
            self.done = 1
            self.desired_state[4] = CoordinatedTurn(self)
            return
        norm1 = self.pos_est[0].item() - self.current_waypoint[0][0].item()
        norm2 = self.pos_est[1].item() - self.current_waypoint[0][1].item()
        norm = np.sqrt(norm1**2 + norm2**2)
        if len(self.waypoints) <= self.current_waypoint[1]:
            self.current_waypoint = ((self.waypoints[1]), 1)
            self.last_waypoint = ((self.waypoints[0]), 0)
            return
        if norm < 50:
            self.last_waypoint = self.current_waypoint
            self.current_waypoint = (self.waypoints[self.current_waypoint[1]], 
                                     self.current_waypoint[1] + 1)
            self.desired_state[4] = guidance(self.pos_est[0].item(),
                                             self.pos_est[1].item(),
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
        self.position_history.append(self.position.copy())
        self.pos_est_hist.append(self.pos_est.copy())
        
    def plot_position(self, **kwargs):
        ostr = kwargs["color"] + "o"
        xstr = kwargs["color"] + "x"
        x = []
        y = []
        for i in range(0, len(self.position_history)):
            x.append(self.position_history[i][0])
            y.append(self.position_history[i][1])
        plt.plot(x, y, 'b-', label='Agent trajectory', linewidth=2)
        plt.plot(self.position_history[0][0], self.position_history[0][1], ostr, label='Initial Start')
        plt.plot(self.position_history[-1][0], self.position_history[-1][1], xstr, label='Initial End')
        plt.xlabel('x-position')
        plt.ylabel('y-position')
    
    def plot_pos_est(self, **kwargs):
        ostr = kwargs["color"] + "o"
        xstr = kwargs["color"] + "x"
        x = []
        y = []
        for i in range(0, len(self.pos_est_hist)):
            x.append(self.pos_est_hist[i][0])
            y.append(self.pos_est_hist[i][1])
        # plt.plot(x, y, ',', label='Cooperative Navigation Solution')
        plt.plot(x, y, 'r--', label='Cooperative Navigation Solution', linewidth=2)
        plt.plot(self.pos_est_hist[0][0], self.pos_est_hist[0][1], ostr, label='Start')
        plt.plot(self.pos_est_hist[-1][0], self.pos_est_hist[-1][1], xstr, label='End')
        plt.xlabel('x-position')
        plt.ylabel('y-position')
    
    def get_meas(self, **kwargs):
        meas = []
        for i in range(0, len(self.obj_pos_true)):
            norm0 = self.obj_pos_true[i][0].item() - self.position[0].item() + \
                    self.position[0].item()
            norm1 = self.obj_pos_true[i][1].item() - self.position[1].item() + \
                    self.position[1].item()
            rg = np.sqrt(norm0**2 + norm1**2)
            bear = -math.atan2(self.obj_pos_true[i][1] - self.position[1] + \
                               self.position[1].item(),
                               self.obj_pos_true[i][0] - self.position[0] + \
                               self.position[0].item())
            meas.append(np.array([[bear], [rg]]))
        self.meas = meas
        self.inverse_meas()
    
    def inverse_meas(self):
        inv_meas = []
        def measfnc(p, bear, rg):
            x, y = p
            return (-math.atan2(y, x) - bear, 
                    np.sqrt(x**2 + y**2) - rg)
        
        for i in range(0, len(self.meas)):
            bear = self.meas[i][0].item()
            rg = self.meas[i][1].item()
            x2, y2 = fsolve(measfnc, (self.obj_pos_true[i][0].item(), 
                                      self.obj_pos_true[i][1].item()), 
                            (bear, rg))
            # x2, y2 = fsolve(measfnc, (self.rfs.states[i][0].item(), 
            #               self.rfs.states[i][2].item()), (bear, rg))
            inv_meas.append(np.array([[x2], [y2]]))
        
        self.inv_meas = inv_meas
    
    def get_object_positions(self, agent_list, num, env):
        self.obj_pos_true = []
        for i in range(0, len(agent_list)):
            if i != num:
                self.obj_pos_true.append(agent_list[i][0].position)
        # for j in range(0, len(env.obstacle_locations)):
        #     obs_pos = np.array([[env.obstacle_locations[j][0].item()], 
        #                         [env.obstacle_locations[j][1].item()]])
        #     self.obj_pos_true.append(obs_pos)
    
    def init_ekf(self):
        def meas_fnc(state, **kwargs):
            mag = state[0, 0]**2 + state[2, 0]**2
            sqrt_mag = np.sqrt(mag)
            mat = np.vstack((np.hstack((state[2, 0] / (mag+1e-3), 0,
                                        -state[0, 0] / (mag+1e-3), 0)),
                            np.hstack((state[0, 0] / (sqrt_mag+1e-3), 0,
                                       state[2, 0] / (sqrt_mag+1e-3), 0))))
            return mat
        
        def meas_mod(state, **kwargs):
            z1 = -np.arctan2(state[2, 0], state[0, 0])
            z2 = np.sqrt(state[0, 0]**2 + state[2, 0]**2)
            return np.array([[z1], [z2]])
        
        kf = filters.ExtendedKalmanFilter()
        kf.set_meas_mat(fnc=meas_fnc)
        kf.set_meas_model(meas_mod)
        kf.meas_noise = np.diag([(10. * np.pi / 180)**2, 10.0**2])
        sig_w = 10.
        G = np.array([[1.**2 / 2, 0],
                      [1.,        0],
                      [0,         1.**2 / 2],
                      [0,         1.]])
        Q = block_diag(sig_w**2 * np.eye(2))
        kf.set_proc_noise(mat=G @ Q @ G.T)
        
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
        
        return kf
        
    def init_cphd(self, kf, env):
        gm = []
        cov = 10.0**2*np.eye(4)
        cov[1, 1] = 5.0**2
        cov[3, 3] = 5.0**2
        for i in range(0, len(self.obj_pos_true)):
            vec = np.array([[self.obj_pos_true[i][0].item()], 
                            [2.], 
                            [self.obj_pos_true[i][1].item()],
                            [2.]])
            gm.append(GaussianMixture(means=[vec], 
                                      covariances=[cov.copy()], weights=[1/len(self.obj_pos_true)]))
    
        self.rfs.save_covs = True
        self.rfs.max_expected_card = len(self.obj_pos_true) + 1
        self.rfs.prob_detection = 0.999
        self.rfs.prob_survive = 0.99
        self.rfs.merge_threshold = 4
        self.rfs.req_births = len(self.obj_pos_true) + 1
        self.rfs.birth_terms = gm
        self.rfs.inv_chi2_gate = 32.2361913029694
        self.rfs.gating_on = False
        self.rfs.clutter_rate = 3.0
        self.rfs.clutter_den = 1 / np.prod(env.pos_bnds[:, [1]] - env.pos_bnds[:, [0]])
        self.rfs.filter = kf
    
    def make_broadcast(self):
        self.broadcast = (self.rfs._gaussMix)
        self.broadcast_hist.append(self.broadcast)
    
    def receive_broadcasts(self, message_list):
        # message_list.append(self.broadcast)
        self.messages = message_list
        self.message_hist.append(message_list)
    
class Target:
    def __init__(self):
        self.position = []
    
    def plot_position(self, **kwargs):
        ostr = kwargs["color"] + "D"
        plt.plot(self.position[0], self.position[1], ostr, label='Target Position')
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
          
        # self.map = rng.random((self.max_x_inds, self.max_y_inds))  # random obstacle locations
        # for ii in range(0, np.size(self.map, axis=0)):
        #     for jj in range(0, np.size(self.map, axis=1)):
        #         if self.map[ii][jj] < obstacle_chance:
        #             self.map[ii][jj] = 1
        #             self.obstacle_list.append((ii, jj))
        #             self.obstacle_locations.append(self.ind_to_pos(ii, jj))
        #         else:
        #             self.map[ii][jj] = 0
        
        self.map = np.zeros((self.max_x_inds, self.max_y_inds))  # set obstacle locations
        # self.map[10, 10:20] = 1
        # self.map[0:14, 3] = 1
        # self.map[20:31, 28] = 1
        # self.map[0:13, 27] = 1
        for ii in range(0, np.size(self.map, axis=0)):
            for jj in range(0, np.size(self.map, axis=1)):
                if self.map[ii][jj] == 1:
                    self.obstacle_list.append((ii, jj))
                    self.obstacle_locations.append(self.ind_to_pos(ii, jj))
        
        self.fix_obstacles()

    def fix_obstacles(self):
        for i in range(0, len(self.start)):
            self.map[self.start[i][0], self.start[i][1]] = 0
        for j in range(0, len(self.end)):
            self.map[self.end[j][0], self.end[j][1]] = 0

    def plot_obstacles(self):
        x_val = [x[0] for x in self.obstacle_locations]
        y_val = [x[1] for x in self.obstacle_locations]
        plt.scatter(x_val, y_val, s=30, marker="s", label='Obstacles', color='black')

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

        self.sysout = sig.StateSpace(F-G@K, G@K, C, D, dt=self.dt)

        self.F = self.sysout.A
        self.G = self.sysout.B

def discretize(dt, A, B, C, D):
    sys = sig.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    F = sysd.A
    G = sysd.B
    return F, G

def CoordinatedTurn(agent):
    phi = np.arctan(agent.u**2/(9.81*75))
    dot = agent.dt * (9.81/agent.u)*np.tan(phi)
    psi = agent.desired_state[4].item() + dot
    return psi

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

def plot_results(agent_list, target_list, env):
    # rfs.plot_card_time_hist(lgnd_loc="lower left", sig_bnd = None) # sig_bnd = None
    # plt.savefig('card_time_hist.pdf', format='pdf',  transparent=True)
    
    # rfs.plot_card_dist()
    
    plt.figure()
    for i in range(0, len(agent_list)):
        a = agent_list[i][0]
        a.plot_position(color="b")
        
    for i in range(0, len(target_list)):
        t = target_list[i]
        t.plot_position(color="g")
        
    for i in range(0, len(agent_list)):
        a = agent_list[i][0]
        a.plot_pos_est(color="r")

    env.plot_obstacles()
    plt.title('True Position vs Navigation Solution')
    plt.grid()
    # plt.savefig('ground_truth.pdf', format='pdf', transparent=True)
    
    # agent = agent_list[0][0]
    # agent.rfs.plot_states([0, 2], state_lbl='Object States', 
    #                       lgnd_loc='lower left', state_color='g', sig_bnd=None)
    # plt.savefig('rfsplot.pdf', format='pdf',  transparent=True)

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
        bear = -math.atan2(y[i][1] - env.pos_bnds[0][1].item(),
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

def init_sim():
    env = Environment()
    obs_chance = 0.2
    x_side = 5000
    y_side = 5000
    env.max_x_inds = 31
    env.max_y_inds = 31
    
    env.world_gen(obs_chance, x_side, y_side)
    
    env.start = [(1, 1),
                 (1, 15),
                 (1, 30),
                 (30, 1),
                 (30, 30)]
    
    randend = 0
    if randend == 0:
        env.end = [(10, 5),
                    (10, 25),
                    (25, 5),
                    (25, 25),
                    (15, 15)]
        # env.end = [(15, 15),
        #             (15, 15),
        #             (15, 15),
        #             (15, 15),
        #             (15, 15)]
    else:
        env.end = []
        for i in range(0, 5):
            env.end.append((int(rng.uniform(1, 30)), int(rng.uniform(1, 30))))

    env.fix_obstacles()
    
    fw  = LateralFixedWing()
    control = LQR(fw.F, fw.G, fw.statedim, fw.indim)
    
    proc_noise = 100.0*np.ones((4, 1))
    proc_noise[1, 0] = 20.0
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
        a.pos_est = a.position.copy()
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
    
    for i in range(0, len(agent_list)):
        agent_list[i][0].get_object_positions(agent_list, agent_list[i][1], env)
        kf = agent_list[i][0].init_ekf()
        agent_list[i][0].init_cphd(kf, env)
    
    return agent_list, target_list, env, kf, control
