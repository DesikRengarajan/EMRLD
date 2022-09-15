Code for Enhanced Meta Reinforcement Learning using Demonstrations in Sparse Reward Environments

This code is based on a public meta-rl github repository https://github.com/learnables/learn2learn/


Libraries to run this code base:
	1. cherry-rl 0.1.4
	2. tensorboard 
	3. learn2learn 
	4. mujoco-py 2.0.2.13
	5. torch 1.10.0
	6. gym 0.21.1

To run the experiments, simply execute the following commands, 

Particle2D Navigation:
	python EMRLD_PN.py    --exp-num i   
	python EMRLD-WS_PN.py --exp-num i

Two Wheeled Locomotion 
	python EMRLD_TW.py    --exp-num i   
	python EMRLD-WS_TW.py --exp-num i

HalfCheetah Forward-Backward
	python EMRLD_HC.py    --exp-num i   
	python EMRLD-WS_HC.py --exp-num i

Where i = 1 for Optimal demonstration data, and i = 2 for sub-optimal demonstration data


