#Python program to create a controller for the Cart-Pole environment v0
#Then train it using reinforcement learning to get an average reward of 200.0

import tensorflow as tf 
import tflearn
import numpy as np 
import gym
from collections import Counter
env = gym.make('MountainCar-v0')

#Function for a random game
def episode(env, param):
	obsv = env.reset()
	tot_reward = 0.0
	for _ in range(goal_steps):
		env.render('True')
		action_space = np.matmul(param, obsv)
		chosen_action = np.argmax(softmax(action_space))
		obsv, reward, done, info = env.step(chosen_action)
		tot_reward += reward 
		if done:
			env.close()
			break
	# env.close()
	return tot_reward

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def initial_population(param, goal_steps, initial_games):
	training_data = []		# [Observations, actions]
	rewards = []	#All scores
	accepted_rewards = []	#Threshold scores
	for episode in range(initial_games):
		score = 0
		obsv = env.reset()
		game_memory = []
		prev_obsv = []
		for _ in range(goal_steps):
			# env.render()
			action_space = np.matmul(param, obsv)
			chosen_action = np.argmax(softmax(action_space))
			obsv, reward, done, info = env.step(chosen_action)

			if len(prev_obsv) > 0:
				game_memory.append([prev_obsv, chosen_action])

			if episode % 500 == 0:
				env.render()

			prev_obsv = obsv
			score += reward
			if done:
				break

		accepted_rewards.append(score)
		for data in game_memory:
			#One-hot
			if data[1] == 2:
				output = [0,0,1]
			if data[1] == 1:
				output = [0,1,0]
			if data[1] == 0:
				output = [1,0,0]
			training_data.append([data[0], output]) 		#Tuple of observations and actions
		# print(training_data)
		env.reset() 
		rewards.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	# print('Average accepted score:', np.mean(accepted_rewards))
	# print('Median score for accepted scores:', np.median(accepted_rewards))
	# print(Counter(accepted_rewards))

	return training_data

def create_model(input_size):
	network = tflearn.layers.core.input_data(shape=[None, input_size, 1], name='inputs')

	# network = tflearn.layers.core.fully_connected(network, 128, activation='relu')
	# network = tflearn.layers.core.dropout(network, 0.8)

	network = tflearn.layers.core.fully_connected(network, 3, activation='softmax')

	network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network)

	return model

def train_env(training_data):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]
	model = create_model(len(X[0]))
	model.fit({'inputs': X},{'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True)
	return	model



if __name__ == '__main__':
	param = np.random.rand(3,2)*2 - 1    #to keep it between -1 and 1
	LR = 1e-3
	goal_steps = 200
	# min_reward = 50
	initial_games = 5000

	training_data = initial_population(param, goal_steps, initial_games)
	# print(training_data[0][0])
	model = train_env(training_data)
	# initial_population(param, goal_steps, min_reward, initial_games)
	# for ep in range(5):
	# 	print('Total reward: {}'.format(episode(env, param)))

	# scores = []
	# choices = []
	# for each_game in range(10):
	# 	score = 0
	# 	game_memory = []
	# 	prev_obs = []
	# 	env.reset()
	# 	for _ in range(goal_steps):
	# 		env.render('True')

	# 		if len(prev_obs)==0:
	# 			action = np.random.choice([0,1])
	# 		else:
	# 			action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

	# 		choices.append(action)
	# 		new_observation, reward, done, info = env.step(action)
	# 		prev_obs = new_observation
	# 		game_memory.append([new_observation, action])
	# 		score += reward
	# 		if done:
	# 			env.close() 
	# 			break
	# 	scores.append(score)

	# print('Average Score:',sum(scores)/len(scores))
	# print('Choice-2: {}  Choice-1: {}  Choice-0: {}'.format(choices.count(2)/len(choices),choices.count(1)/len(choices),choices.count(0)/len(choices)))
	# # print(min_reward)