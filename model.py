"""
아래 문제로 걍 pytorch로 변경
https://github.com/tensorflow/tensorflow/issues/35376
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import loggers as logging
from settings import run_folder, run_archive_folder
import mcts_config as config

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenModel:
	def __init__(self, reg_const, learning_rate, input_dim, output_dim):
		self.reg_const = reg_const
		self.learning_rate = learning_rate
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_layers = config.HIDDEN_CNN_LAYERS
		# todo : find out channel num!!
		self.model = Network(self.hidden_layers, input_dim, output_dim, config.INPUT_CHANNEL).to(config.device)
		self.optim = torch.optim.SGD(self.model.parameters(), lr= learning_rate, momentum=config.MOMENTUM)
		self.loss_weight = {'value_head': 0.5, 'policy_head': 0.5}

	def predict(self, x):
		v, p = self.model.forward(x)
		return v.cpu().detach().numpy(), p.cpu().detach().numpy()

	def update(self, states, targets):
		"""
		:param states:   state transformed by convertToModelInput
		:param targets:  { 'value_head' : value ndrray, 'policy_head' : policy ndarray}
		:return:
		"""
		assert states.shape[0] == config.BATCH_SIZE
		assert targets.shape[0] == config.BATCH_SIZE

		# todo : calculate loss with MSE for value_head and with softmax_cross_entropy_with_logits for policy_head!!
		# todo : add parameter for spliting validation and verbose, epcochs
		predict_v, predict_p = self.model(states).numpy()

		loss = torch.tensor(0.).to(config.device)

		# softmax_cross_entropy_with_logits for policy_head
		p_loss = F.softmax(predict_p, dim=-1)
		p_loss = -torch.sum(targets['policy_head'] * torch.log(p_loss), 1)
		p_loss = torch.unsqueeze(loss, 1)

		loss += self.loss_weight['policy_head'] * torch.mean(p_loss)

		# loss with MSE for value_head
		v_loss = F.mse_loss(predict_p, targets['value_head'])
		loss += self.loss_weight['value_head'] * p_loss

		# optimize the model
		self.optim.zero_grad()
		loss.backward()

		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optim.step()

	def convert_to_model_input(self, state):
		inputToModel = state.binary  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
		inputToModel = np.reshape(inputToModel, self.input_dim)
		inputToModel = torch.tensor(inputToModel, dtype=torch.float32).to(config.device)
		return (inputToModel)

	def write(self, game, version):
		torch.save(self.model.state_dict(), run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

	def read(self, game, run_number, version):
		# todo : check softmax_cross_entropy_with_logits. In original code, it use load_model api from tensorflow passing the custom object.
		self.model.load_state_dict(
			torch.load(
				run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.h5'))
		self.model.eval()


class ConvLayer(nn.Module):

	def __init__(self, in_channel, n_filters, kernel_size, padding, use_relu=True):
		super(ConvLayer, self).__init__()
		self.in_channel = in_channel
		self.n_filters = n_filters
		self.kernel_size = kernel_size
		self.use_relu = use_relu
		self.padding = padding
		# https://ezyang.github.io/convolution-visualizer/index.html
		# o = [i + 2 * p - k] / s + 1
		# (o + k) / 2  = p   o = i
		self.conv2d = nn.Conv2d(in_channel, n_filters, kernel_size, padding=padding, bias=False)
		if use_relu:
			self.leaky_relu = nn.LeakyReLU()
		self.bn_2d = nn.BatchNorm2d(n_filters)
		# todo : add l2 regularizer with reg_const (weight_decay) in adam optimizer

	def forward(self, x):
		x = self.conv2d(x)

		if self.padding == 2:
			x = x[:,:,:-1,:-1]

		x = self.bn_2d(x)
		if self.use_relu:
			x = self.leaky_relu(x)
		return x


class RedisualLayer(nn.Module):

	def __init__(self, in_channel, n_filters, kernel_size):
		super(RedisualLayer, self).__init__()

		self.conv2d = ConvLayer(in_channel, n_filters, kernel_size, 2)
		self.conv2d_residual = ConvLayer(in_channel, n_filters, kernel_size, 2, False)
		self.leaky_relu_final = nn.LeakyReLU()

	def forward(self, x):
		residual = self.conv2d(x)
		x = self.conv2d_residual(x)
		x = torch.cat([residual, x])
		x = self.leaky_relu_final(x)
		return x


class ValueHeadNetwork(nn.Module):
	def __init__(self, in_channel):
		super(ValueHeadNetwork, self).__init__()
		self.conv2d = ConvLayer(in_channel, 1, (1,1), 0)
		self.bn2d = nn.BatchNorm2d(1)
		self.leacky_relu = nn.LeakyReLU()
		# todo : check : input_dim
		self.linear_20 = nn.Linear(42, 20, bias=False)
		self.linear_1 = nn.Linear(20, 1, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.conv2d(x)
		x = self.bn2d(x)
		x = self.leacky_relu(x)
		x = torch.flatten(x, start_dim=2)
		x = self.linear_20(x)
		x = self.leacky_relu(x)
		x = self.linear_1(x)
		x = self.tanh(x)
		return x


class PolicyHeadNetwork(nn.Module):
	def __init__(self, in_channel, output_dim):
		super(PolicyHeadNetwork, self).__init__()

		self.in_channel = in_channel
		self.output_dim = output_dim

		self.conv2d = ConvLayer(in_channel, 2, (1,1), 0)
		self.bn2d = nn.BatchNorm2d(2)
		self.leacky_relu = nn.LeakyReLU()
		self.linear = nn.Linear(84, output_dim, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.conv2d(x)
		x = self.bn2d(x)
		x = self.leacky_relu(x)
		x = torch.flatten(x, start_dim=1)
		x = self.linear(x)
		return x


class Network(nn.Module):
	def __init__(self, hidden_layers, input_dim, output_dim, in_channel):
		super(Network, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.in_channel = in_channel

		self.hidden_layers = hidden_layers
		self.num_layers = len(hidden_layers)

		self.start_n_filters = self.hidden_layers[0]['filters']
		self.start_kernel_size = self.hidden_layers[0]['kernel_size']
		self.conv2d = ConvLayer(in_channel, self.start_n_filters, self.start_kernel_size, 2)

		tmp_in_channel = self.start_n_filters

		if len(self.hidden_layers) > 1:
			modules = []
			for h in self.hidden_layers[1:]:
				modules.append(RedisualLayer(tmp_in_channel, h['filters'], h['kernel_size']))
				tmp_in_channel = h['filters']
			self.residuals = nn.ModuleList(modules)

		self.policy_head = PolicyHeadNetwork(tmp_in_channel, output_dim)
		self.value_head = ValueHeadNetwork(tmp_in_channel)

	def forward(self, x):
		# assert x.shape == self.input_dim
		x.unsqueeze_(0)
		x = self.conv2d(x)

		for l in self.residuals:
			x = l(x)

		vh = self.value_head(x)
		ph = self.policy_head(x)

		return vh, ph