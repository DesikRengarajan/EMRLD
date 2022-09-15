#!/usr/bin/env python3

"""
EMRLD
"""

import random
from copy import deepcopy
import pickle
import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import matplotlib.pyplot as plt
import learn2learn as l2l
from policies import DiagNormalPolicy
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import argparse
from envs.particle_2d import Particles2DEnv


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
	# Update baseline
	returns = ch.td.discount(gamma, rewards, dones)
	baseline.fit(states, returns)
	values = baseline(states)
	next_values = baseline(next_states)
	bootstraps = values * (1.0 - dones) + next_values * dones
	next_value = torch.zeros(1, device=values.device)
	return ch.pg.generalized_advantage(tau=tau,
									   gamma=gamma,
									   rewards=rewards,
									   dones=dones,
									   values=bootstraps,
									   next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau, task_data, w_a2c, w_bc):
	# Update policy and baseline
	demo_states = torch.tensor(task_data['state']).float()
	demo_actions = torch.tensor(task_data['action']).float()
	demo_adv = torch.ones((demo_states.shape[0], 1))
	demo_log_probs = learner.log_prob(demo_states, demo_actions)
	if train_episodes is None:
		bc_loss = a2c.policy_loss(demo_log_probs, demo_adv)
		return bc_loss
	else:
		states = train_episodes.state()
		actions = train_episodes.action()
		rewards = train_episodes.reward()
		dones = train_episodes.done()
		next_states = train_episodes.next_state()
		advantages = compute_advantages(baseline, tau, gamma, rewards,
										dones, states, next_states)
		advantages = ch.normalize(advantages).detach()
		log_probs = learner.log_prob(states, actions)
		bc_loss_2 = a2c.policy_loss(demo_log_probs, w_bc * demo_adv)
		a2c_loss = a2c.policy_loss(log_probs, w_a2c * advantages)
		return  bc_loss_2 + a2c_loss



def fast_adapt_a2c(clone, train_episodes, adapt_lr,adapt_a2c_lr, baseline, gamma, tau, task_data, w_a2c, w_bc,
				   first_order=False):
	if ((train_episodes is  not None) and (adapt_a2c_lr > 0)):
		adapt_lr = adapt_a2c_lr
	second_order = not first_order
	loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau, task_data, w_a2c, w_bc)
	gradients = autograd.grad(loss,
							  clone.parameters(),
							  retain_graph=second_order,
							  create_graph=second_order)
	return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr, adapt_a2c_lr,traj_data,
						iteration, w_a2c, w_bc):
	mean_loss = 0.0
	mean_kl = 0.0
	task_id = 0
	for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
										 total=len(iteration_replays),
										 desc='Surrogate Loss',
										 leave=False):
		train_replays = task_replays[:-1]
		valid_episodes = task_replays[-1]
		new_policy = l2l.clone_module(policy)
		task_data = traj_data[task_id]
		task_id += 1
		# Fast Adapt
		for train_episodes in train_replays:
			new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,adapt_a2c_lr,
										baseline, gamma, tau, task_data, w_a2c, w_bc, first_order=False)


			# Useful values
		states = valid_episodes.state()
		actions = valid_episodes.action()
		next_states = valid_episodes.next_state()
		rewards = valid_episodes.reward()
		dones = valid_episodes.done()

		# Compute KL
		old_densities = old_policy.density(states)
		new_densities = new_policy.density(states)
		kl = kl_divergence(new_densities, old_densities).mean()
		mean_kl += kl

		# Compute Surrogate Loss
		advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
		advantages = ch.normalize(advantages).detach()
		old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
		new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
		mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
	mean_kl /= len(iteration_replays)
	mean_loss /= len(iteration_replays)
	return mean_loss, mean_kl


def main(
		env_name='Particles2DEnv',
		adapt_lr=0.1,
		meta_lr=1.0,
		adapt_steps = 2,
		num_iterations=500,
		meta_bsz=24,
		adapt_bsz=20,
		tau=1.00,
		gamma=0.95,
		seed=42,
		num_workers=10,
		cuda=0,
		gpu_index=0,
		tresh=0.02,
		is_sparse=False,
		r_tresh=0.2,
		w_a2c=1.0,
		w_bc=1.0,
		data_path='',
		load=False,
		policy_path='',
		baseline_path='',
		adapt_a2c_lr=-1
):
	cuda = bool(cuda)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	device_name = 'cpu'
	log_path = 'Results/EMRLD/{}/meta_rl_{}'.format(env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	print("Running EMRLD for Particle2D Navigation")
	writer = SummaryWriter(log_path)
	writer.add_text('env_name', str(env_name))
	writer.add_text('adapt_lr', str(adapt_lr))
	writer.add_text('meta_lr', str(meta_lr))
	writer.add_text('num_iterations', str(num_iterations))
	writer.add_text('adapt_bsz', str(adapt_bsz))
	writer.add_text('adapt_steps', str(adapt_steps))
	writer.add_text('num_workers', str(num_workers))
	writer.add_text('seed', str(seed))
	writer.add_text('tresh', str(tresh))
	writer.add_text('r_tresh', str(r_tresh))
	writer.add_text('w_a2c', str(w_a2c))
	writer.add_text('w_bc', str(w_bc))
	writer.add_text('Data', data_path)
	if cuda:
		torch.cuda.manual_seed(seed)
		device_name = 'cuda'
	device = torch.device(device_name)

	def make_env():
		env = Particles2DEnv(tresh = tresh,r_tresh = r_tresh ,is_sparse=True,project_circular = True)
		env = ch.envs.ActionSpaceScaler(env)
		return env

	env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
	env.seed(seed)
	env.set_task(env.sample_tasks(1)[0])
	env = ch.envs.Torch(env)
	policy = DiagNormalPolicy(env.state_size, env.action_size, device=device)
	if cuda:
		policy = policy.to(device)
	baseline = LinearValue(env.state_size, env.action_size)
	traj_data = pickle.load(open(data_path, 'rb'))
	meta_bsz = len(traj_data)
	writer.add_text('meta_bsz', str(meta_bsz))
	print('Meta Batch Size: ',meta_bsz)    
	# for logging
	net_adaptation_reward = []
	adapt_time = []
	meta_update_time = []
	if load:
		policy.load_state_dict(torch.load(policy_path))
		baseline.load_state_dict(torch.load(baseline_path))

	for iteration in range(num_iterations):
		t0 = time.time()
		iteration_reward = 0.0
		bc_reward = 0.0
		iteration_replays = []
		iteration_policies = []
		test_episodes = []
		task_config_list = []
		for task_config in tqdm(traj_data, leave=False, desc='Data'):  

			task_data = traj_data[task_config]
			clone = deepcopy(policy)
			goal_task = {'goal': np.array([task_data['goal_loc'][0], task_data['goal_loc'][1]])}
			env.set_task(goal_task)
			env.reset()
			task = ch.envs.Runner(env)
			task_replay = []            

			for step in range(adapt_steps):
				### RL-BC Adapt Step ###
				train_episodes = task.run(clone, episodes=adapt_bsz)
				if cuda:
					train_episodes = train_episodes.to(device, non_blocking=True)
				task_replay.append(train_episodes)                               
				clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,adapt_a2c_lr,
									   baseline, gamma, tau, task_data, w_a2c, w_bc,
									   first_order=True)


			valid_episodes = task.run(clone, episodes=adapt_bsz)
			task_replay.append(valid_episodes)
			iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
			iteration_replays.append(task_replay)
			iteration_policies.append(clone)

		# Print statistics
		t1 = time.time()
		print('\nIteration', iteration)
		adaptation_reward = iteration_reward / meta_bsz
		print('adaptation_reward', adaptation_reward)        
		writer.add_scalar('avg adaptation reward', adaptation_reward, iteration + 1)
		adapt_time.append(t1 - t0)
		net_adaptation_reward.append(adaptation_reward)


		# TRPO meta-optimization
		backtrack_factor = 0.5
		ls_max_steps = 15
		max_kl = 0.01
		if cuda:
			policy = policy.to(device, non_blocking=True)
			baseline = baseline.to(device, non_blocking=True)
			iteration_replays = [[r.to(device, non_blocking=True) for r in task_replays] for task_replays in
								 iteration_replays]

		# Compute CG step direction
		old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma,
											   adapt_lr,adapt_a2c_lr, traj_data, iteration, w_a2c, w_bc)
		grad = autograd.grad(old_loss,
							 policy.parameters(),
							 retain_graph=True)
		grad = parameters_to_vector([g.detach() for g in grad])
		Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
		step = trpo.conjugate_gradient(Fvp, grad)
		shs = 0.5 * torch.dot(step, Fvp(step))
		lagrange_multiplier = torch.sqrt(shs / max_kl)
		step = step / lagrange_multiplier
		step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
		vector_to_parameters(step, step_)
		step = step_
		del old_kl, Fvp, grad
		old_loss.detach_()

		# Line-search
		for ls_step in range(ls_max_steps):
			stepsize = backtrack_factor ** ls_step * meta_lr
			clone = deepcopy(policy)
			for p, u in zip(clone.parameters(), step):
				p.data.add_(-stepsize, u.data)
			new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, tau, gamma,
											   adapt_lr,adapt_a2c_lr, traj_data, iteration, w_a2c, w_bc)
			if new_loss < old_loss and kl < max_kl:
				for p, u in zip(policy.parameters(), step):
					p.data.add_(-stepsize, u.data)
				break

		t2 = time.time()
		meta_update_time.append(t2 - t1)
		print('Time per iteration: ',t2 - t0)

	env.close()
	return 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='EMRLD')
	parser.add_argument('--tresh', type=float, default=0.02, metavar='G',
						help='Goal Treshold')
	parser.add_argument('--r-tresh', type=float, default=0.2, metavar='G',
						help='Sparse Reward Treshold')
	parser.add_argument('--adapt-lr', type=float, default=0.01, metavar='G',
						help='adaptation learning rate')
	parser.add_argument('--adapt-a2c-lr', type=float, default=-1, metavar='G',
						help='a2c adaptation learning rate')
	parser.add_argument('--w-a2c', type=float, default=0.2, metavar='G',
						help='w-rl')
	parser.add_argument('--w-bc', type=float, default=1.0, metavar='G',
						help='w-bc')
	parser.add_argument('--data-path', metavar='G', default='Traj_Data/Good_PN.p',
						help='path of data')
	parser.add_argument('--workers', type=int, default=15, metavar='N',
						help='Number of workers')
	parser.add_argument('--adapt-steps', type=int, default=1, metavar='N',
						help='Adapt Steps')
	parser.add_argument('--seed', type=int, default=42, metavar='N',
						help='Seed')
	parser.add_argument('--load', action='store_true', default=False,
						help='Load policy')
	parser.add_argument('--max-iter', type=int, default=700, metavar='N',
						help='Max iter')
	parser.add_argument('--exp-num', type=int, default=1, metavar='N',
						help='1: Good 2:Bad')
	args = parser.parse_args()
	args.policy_path = ''
	args.baseline_path = ''

	if args.exp_num == 1:
		args.data_path = 'Traj_Data/Good_PN.p'
	elif args.exp_num == 2:
		args.data_path = 'Traj_Data/Bad_PN.p'
		args.w_a2c = 1.0
	else:
		print("Running using non default values")

	a = main(tresh=args.tresh,
			 r_tresh=args.r_tresh,
			 w_a2c=args.w_a2c,
			 w_bc=args.w_bc,
			 data_path=args.data_path,
			 num_workers=args.workers,
			 load=args.load,
			 policy_path=args.policy_path,
			 baseline_path=args.baseline_path,
			 adapt_steps=args.adapt_steps,
			 adapt_lr=args.adapt_lr,
			 adapt_a2c_lr=args.adapt_a2c_lr,
			 seed=args.seed,
			 num_iterations=args.max_iter)