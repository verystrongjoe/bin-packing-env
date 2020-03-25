import numpy as np
import random
import MCTS as mc
from game import GameState

import mcts_config as config
import logging
import time

import matplotlib.pyplot as plt
# import pylab as pl


class User:
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)


class Agent:
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size
		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model
		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []


	def simulate(self):
		logging.info('ROOT NODE ID : %s', self.mcts.root.state.id)
		self.mcts.root.state.render(logging)   # GameState
		logging.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		# Move the leaf node
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(logging)  # GameState

		# Evaluate the leaf node
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		# Backfill the value through the tree
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		# run the simulation
		for sim in range(self.MCTSsimulations):
			logging.info('***************************')
			logging.info('****** SIMULATION %d ******', sim + 1)
			logging.info('***************************')
			self.simulate()

		# get action values
		pi, values = self.getAV(1)

		# pick the action
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)

		NN_value = -self.get_preds(nextState)[0]

		logging.info('ACTION VALUES...%s', pi)
		logging.info('CHOSEN ACTION...%d', action)
		logging.info('MCTS PERCEIVED VALUE...%f', value)
		logging.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):

		# predict the leaf
		inputToModel = self.model.convert_to_model_input(state)

		preds = self.model.predict(inputToModel)

		value_array = preds[0]
		logits_array = preds[1]
		value = value_array[0]

		logits = logits_array[0]
		allowedActions = state.allowedActions

		mask = np.ones(logits.shape,dtype=bool)
		mask[allowedActions] = False
		logits[mask] = -100

		# softmax
		odds = np.exp(logits)
		probs = odds / np.sum(odds)  # put this just before the for?

		return ((value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		logging.info('------EVALUATING LEAF------')

		if done == 0:
	
			value, probs, allowedActions = self.get_preds(leaf.state)
			logging.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

			probs = probs[allowedActions]

			for idx, action in enumerate(allowedActions):
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					self.mcts.addNode(node)
					logging.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					logging.info('existing node...%s...', node.id)

				newEdge = mc.Edge(leaf, node, probs[idx], action)
				leaf.edges.append((action, newEdge))
				
		else:
			logging.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau)
			values[action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		if tau == 0:  # deterministic
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else: # stochastic
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		return action, value

	def replay(self, ltmemory):
		logging.info('******RETRAINING MODEL******')

		import torch
		for i in range(config.TRAINING_LOOPS):
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			training_states = torch.stack([self.model.convert_to_model_input(row['state']) for row in minibatch])
			training_targets = {'value_head': np.array([row['value'] for row in minibatch])
								, 'policy_head': np.array([row['AV'] for row in minibatch])} 

			loss = self.model.update(training_states, training_targets)
			# fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)

			logging.info('NEW LOSS %s', loss)

			# logging.info('NEW LOSS %s', fit.history)
			# self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			# self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4))
			# self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4))

		# todo : display and find out how it works
		plt.plot(self.train_overall_loss, 'k')
		plt.plot(self.train_value_loss, 'k:')
		plt.plot(self.train_policy_loss, 'k--')
		plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
		# display.clear_output(wait=True)
		# display.display(pl.gcf())
		# pl.gcf().clear()
		# time.sleep(1.0)
		print('\n')
		# self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict(inputToModel).numpy()
		return preds

	def buildMCTS(self, state):
		logging.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		logging.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]