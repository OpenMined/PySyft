import pytest
import torch as th
import torch.nn as nn

import syft as sy
import syft.frameworks.torch.nn as syft_nn
from syft.execution.plan import Plan
from syft.frameworks.torch.nn import RNNBase


# @pytest.fixture(scope="function", autouse=True)
def test_rnn_plan_example():
	"""
	Prepares simple static federated learning training plan example that use an RNN.
	"""
	# Disable translators
	Plan._build_translators = []

	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.vocab_size = 1000
			emb_size = 100
			num_hidden = 200
			num_rnn_layers = 2
			self.encoder = nn.Embedding(self.vocab_size, emb_size)
			self.rnn = syft_nn.GRU(emb_size, num_hidden, num_rnn_layers)
			# FIX If `bias=True`, then the decoder returns `None`.
			self.decoder = nn.Linear(num_hidden, self.vocab_size, bias=False)

		def init_hidden(self, batch_size):
			# FIX This method shouldn't really be needed since the Syft classes call their own implementation when needed.
			seq_len = 5
			# Make some fake input.
			inp = th.rand(seq_len, batch_size, 1)
			# FIX The `rnn._init_hidden` assumes that the input has a `has_child` method.
			if not hasattr(inp, 'has_child'):
				def has_child(self):
					return hasattr(self, 'child') and self.child is not None

				type(inp).has_child = has_child
			return self.rnn._init_hidden(inp)

		def forward(self, x, hidden=None):
			emb = self.encoder(x)
			# FIX Corrections for PySyft.
			if hasattr(emb, 'child'):
				emb.device = emb.child.child.device
				emb.dtype = emb.child.child.dtype
			if not hasattr(emb, 'has_child'):
				def has_child(self):
					return hasattr(self, 'child') and self.child is not None

				type(emb).has_child = has_child
			output, hidden = self.rnn(emb, hidden)

			decoded = self.decoder(output)
			# FIX
			assert decoded is not None, "The result of the decoder is None. This can happen if the decoder uses a bias."
			# FIXME Find a way to reshape that will work with PySyft.
			# `view` and `flatten` both yield params with grad=None.
			# decoded = decoded.view(-1, self.vocab_size)
			# decoded = decoded.flatten(0, 1)

			# FIX 'ReshapeBackward' object has no attribute 'self_'
			decoded = decoded.reshape(-1, self.vocab_size)
			return decoded, hidden

	model = Net()

	def set_model_params(module, params_list, start_param_idx=0):
		""" Set params list into model recursively
		"""
		param_idx = start_param_idx

		for name, param in module._parameters.items():
			if param is not None:
				module._parameters[name] = params_list[param_idx]
				param_idx += 1

		for name, child in module._modules.items():
			if child is not None:
				param_idx = set_model_params(child, params_list, param_idx)

		return param_idx

	def softmax_cross_entropy_with_logits(logits, targets, batch_size):
		# numstable logsoftmax
		norm_logits = logits - logits.max()
		log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
		return -(targets * log_probs).sum() / batch_size

	def naive_sgd(param, **kwargs):
		assert param.grad is not None, f"A param has a grad that is None: {param}"
		return param - kwargs["lr"] * param.grad

	@sy.func2plan()
	def train(data, initial_hidden, targets, lr, batch_size, model_parameters):
		# load model params
		set_model_params(model, model_parameters)

		# forward
		# FIX Ideally we should have to pass the initial hidden state since a default can be determined but it causes issues when creating a Plan.
		logits, hidden = model(data, initial_hidden)

		# loss
		loss = softmax_cross_entropy_with_logits(logits, targets, batch_size)

		# backward
		loss.backward()

		# step
		updated_params = [naive_sgd(param, lr=lr) for param in model_parameters]

		# accuracy
		pred = th.argmax(logits, dim=1)
		targets_idx = th.argmax(targets, dim=1)
		acc = pred.eq(targets_idx).sum().float() / batch_size

		return (loss, acc, *updated_params)

	# Dummy inputs
	batch_size = 3
	sequence_length = 5
	vocab_size = model.vocab_size
	# Data has the index of the word in a vocabulary.
	data = th.randint(0, vocab_size, (sequence_length, batch_size))
	initial_hidden = model.init_hidden(batch_size)
	target = th.randint(0, vocab_size, (sequence_length * batch_size,))
	target = nn.functional.one_hot(target, vocab_size)
	lr = th.tensor([0.1])
	batch_size = th.tensor([float(batch_size)])
	model_state = list(model.parameters())

	# FIX The default `_apply_time_step` has a bug. See details below.
	def _apply_time_step(self, x, hc, t, reverse_direction=False):
		"""
		Apply RNN layers at time t, given input and previous hidden states
		"""
		rnn_layers = self.rnn_backward if reverse_direction else self.rnn_forward

		# FIX Start
		if len(hc) == 1:
			# The GRU case falls into here.
			hc = th.unsqueeze(hc[0], dim=0)
		else:
			# FIX torch.stack doesn't work with PySyft.
			hc = th.stack([*hc])
		hc_next = th.zeros_like(hc)
		# FIX End

		for layer in range(self.num_layers):
			inp = x[t, :, :] if layer == 0 else hc_next[0][layer - 1, :, :].clone()

			if self.is_lstm:
				hc_next[:, layer, :, :] = th.stack(rnn_layers[layer](inp, hc[:, layer, :, :]))
			else:
				hc_next[0][layer, :, :] = rnn_layers[layer](inp, hc[0][layer, :, :])

		return hc_next

	RNNBase._apply_time_step = _apply_time_step

	# Build Plan
	train.build(data, initial_hidden, target, lr, batch_size, model_state, trace_autograd=True)

	return train, data, initial_hidden, target, lr, batch_size, model_state
