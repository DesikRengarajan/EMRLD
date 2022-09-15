"""
Differential_drive_model
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time
import itertools
import argparse
import datetime
import random
import torch
from collections import defaultdict
from learn2learn.gym.envs.meta_env import MetaEnv


MAX_STEER = 2.84
MAX_SPEED = 0.22
MIN_SPEED = 0.
MAX_ESP_LEN = 100
MAX_GRID = 2.5
# Vehicle parameters
LENGTH = 0.25  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.05  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.15  # [m]

show_animation = True

class TwoWheelEnv(MetaEnv):

	def __init__(self, tresh = 0.05,r_tresh = 0.2,is_relative = False,is_sparse = False,task=None):
		self.seed()
		super(TwoWheelEnv,self).__init__()		
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([0., -2.84]), np.array([0.22, 2.84]), dtype = np.float32) 
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
											shape=(3,), dtype=np.float32)
		self.tresh = tresh
		self.r_tresh = r_tresh
		self.is_sparse = is_sparse
		self.num_step = 0
		self.goal_set = False
		self.is_relative = is_relative
		self.traj_x = []
		self.traj_y = []


	# -------- MetaEnv Methods --------
	def sample_tasks(self, num_tasks):
		"""
		Tasks correspond to a goal point chosen uniformly at random.
		"""
		goals = np.random.uniform(-0.5, 0.5, size=(num_tasks, 2))
		tasks = [{'goal': goal} for goal in goals]
		return tasks

	def set_task(self, task):
		self._task = task
		self.goal_set = True
		self.target = task['goal']



	def reset(self):		
		self.num_step = 0
		if self.is_relative:
			self.pose = np.array([0., 0.,0.])
			r = 2. 
			if self.goal_set is False:
				theta = np.pi*np.random.random()
				self.target[0] = r * np.cos(theta)
				self.target[1] = r * np.sin(theta)
			head_to_target = self.get_heading(self.pose, self.target)
			x = self.pose[0] - self.target[0]
			y = self.pose[1] - self.target[1]
			z = self.pose[2] - head_to_target
			return np.array([x, y, z])
		else:
			self.pose = np.array([0., 0.,0.])

			return self.pose	

	def update_state(self, state, a, DT):
		lin_velocity = a[0]
		rot_velocity = a[1]

		state[0] = state[0] + lin_velocity * math.cos(state[2]) * DT
		state[1] = state[1] + lin_velocity * math.sin(state[2]) * DT
		state[2] = state[2] + rot_velocity * DT

		return state

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))



	def step(self,action):
		self.num_step += 1
		reward = 0
		done = False
		info = {}
		self.action = [max(0,min(0.22,action[0])),max(-2.84,min(2.84,action[1]))]
		self.pose = self.update_state(self.pose, self.action, 0.5) 
		head_to_target = self.get_heading(self.pose, self.target)
		x = self.pose[0] - self.target[0]
		y = self.pose[1] - self.target[1]
		z =  self.pose[2] - head_to_target
		
		reward = -np.sqrt(x ** 2 + y ** 2)
		done = ( ((np.abs(x) < self.tresh) and (np.abs(y) < self.tresh)) or (self.num_step > MAX_ESP_LEN) or (abs(self.pose[0]) > MAX_GRID) or (abs(self.pose[1]) > MAX_GRID) )		

		info['Goal'] = ((np.abs(x) < self.tresh) and (np.abs(y) < self.tresh))
		if self.is_sparse:
			if ((np.abs(x) < self.tresh) and (np.abs(y) < self.tresh)):
				reward = 100. - self.num_step
			elif  (np.sqrt(x ** 2 + y ** 2) < self.r_tresh):
				reward =  2 - np.sqrt(x ** 2 + y ** 2) 
			else:
				reward = 1e-5

		if self.is_relative:
			return np.array([x, y,z]), reward, done, info
		else:
			return self.pose, reward, done, self._task


	

	def render(self, mode="human"):
		self.traj_x.append(self.pose[0])
		self.traj_y.append(self.pose[1])
	  
		plt.cla()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		plt.plot(self.traj_x, self.traj_y, "ob", markersize = 2, label="trajectory")
		plt.plot(self.target[0], self.target[1], "xg", label="target")
		self.plot_car()
		plt.axis("equal")
		plt.grid(True)
		plt.title("Simulation")
		plt.pause(0.0001)
		
	def close(self):
		pass



	def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
		# print("Plotting Car")
		x = self.pose[0]
		y = self.pose[1]
		yaw = self.pose[2] 
		steer = self.action[1]

		outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
							[WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

		fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
							 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

		rr_wheel = np.copy(fr_wheel)

		fl_wheel = np.copy(fr_wheel)
		fl_wheel[1, :] *= -1
		rl_wheel = np.copy(rr_wheel)
		rl_wheel[1, :] *= -1

		Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
						 [-math.sin(yaw), math.cos(yaw)]])
		Rot2 = np.array([[math.cos(steer), math.sin(steer)],
						 [-math.sin(steer), math.cos(steer)]])

		fr_wheel = (fr_wheel.T.dot(Rot2)).T
		fl_wheel = (fl_wheel.T.dot(Rot2)).T
		fr_wheel[0, :] += WB
		fl_wheel[0, :] += WB

		fr_wheel = (fr_wheel.T.dot(Rot1)).T
		fl_wheel = (fl_wheel.T.dot(Rot1)).T

		outline = (outline.T.dot(Rot1)).T
		rr_wheel = (rr_wheel.T.dot(Rot1)).T
		rl_wheel = (rl_wheel.T.dot(Rot1)).T

		outline[0, :] += x
		outline[1, :] += y
		fr_wheel[0, :] += x
		fr_wheel[1, :] += y
		rr_wheel[0, :] += x
		rr_wheel[1, :] += y
		fl_wheel[0, :] += x
		fl_wheel[1, :] += y
		rl_wheel[0, :] += x
		rl_wheel[1, :] += y

		plt.plot(np.array(outline[0, :]).flatten(),
				 np.array(outline[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fr_wheel[0, :]).flatten(),
				 np.array(fr_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fl_wheel[0, :]).flatten(),
				 np.array(fl_wheel[1, :]).flatten(), truckcolor)
		plt.plot(x, y, "*")


