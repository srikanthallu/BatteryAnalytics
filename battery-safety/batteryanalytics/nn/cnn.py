import inspect

import numpy as np
import sklearn
import sklearn.base
import sklearn.model_selection

import torch
import torch.nn
import torch.optim
import torch.utils.data

from .datasets import SubsequenceDataset
from .train import train_model

class View(torch.nn.Module):
	def __init__(self, *shape):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)

	def forward(self, X):
		#print("VIEW forward", X.shape, self.shape)
		return torch.reshape(X, self.shape)

	def extra_repr(self):
		return f"shape={self.shape}"

class _CNN(torch.nn.Module):
	def __init__(self, input_shape, output_shape, n_filters=1, kernel_size=2, hidden_dim=None, transfer=None):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)
		in_channels, out_channels, kernel_sizes, dims, transfer = _CNN._validate_parameters(input_shape, output_shape, n_filters, kernel_size, hidden_dim, transfer)
		self._model = _CNN._get_model(in_channels, out_channels, kernel_sizes, dims, transfer)

	def forward(self, X):
		#print(self._model)
		#print( torch.transpose(X, 1, 2).shape )
		return self._model(
			torch.transpose(X, 1, 2)
		).reshape((-1,) + self.output_shape)

	@staticmethod
	def _validate_parameters(input_shape, output_shape, n_filters, kernel_size, hidden_dim, transfer):
		n_filters = _CNN._validate_n_filters(n_filters)
		kernel_size = _CNN._validate_kernel_size(kernel_size)
		hidden_dim = _CNN._validate_hidden_dim(hidden_dim)
		transfer = _CNN._validate_transfer(transfer)

		smaller,larger = (n_filters,kernel_size) if len(n_filters)<=len(kernel_size) else (kernel_size,n_filters)
		if len(smaller) == 1 and len(larger) > 1:
			smaller.extend( smaller * (len(larger)-1) )
		if len(n_filters) != len(kernel_size):
			raise ValueError(
				f"number of convolutional layers specified by n_filters ({len(n_filters)}) "
				f"does not match number of convolutional layers specified by kernel_size ({len(kernel_size)})"
			)

		# in_channels, out_channels, kernel_size
		in_channels = [input_shape[1]] + n_filters[:-1]
		out_channels = n_filters

		#print("n_filters", n_filters)
		#print("input_shape", input_shape)
		#print("kernel_size", kernel_size)
		input_dim = n_filters[-1] * (input_shape[0] - np.sum(kernel_size) + len(kernel_size))
		#print(f"{n_filters[-1]} * ({input_shape[0]} - {np.sum(kernel_size)} + {len(kernel_size)})")
		#print("input_dim", input_dim)
		output_dim = np.prod(output_shape)
		dims = [input_dim] + hidden_dim + [output_dim]
		if len(transfer) >= len(dims):
			raise ValueError(
				f"number of transfer functions ({len(transfer)}) exceeds "
				f"number of layers ({len(dims)-1})"
			)
		elif len(transfer) == 1:
			if len(dims) > 2:
				transfer.extend( transfer * (len(dims)-3) )
				transfer.append(None)
		elif len(transfer)+2 == len(dims):
			transfer.append(None)
		elif len(transfer)+1 != len(dims):
			raise ValueError(
				f"number of transfer functions ({len(transfer)}) "
				f"does not match number of layers ({len(dims)-1})"
			)
		return in_channels, out_channels, kernel_size, dims, transfer
	
	@staticmethod
	def _validate_n_filters(n_filters):
		if not hasattr(n_filters, "__iter__"):
			return [n_filters]
		else:
			return list(n_filters)
	
	@staticmethod
	def _validate_kernel_size(kernel_size):
		if not hasattr(kernel_size, "__iter__"):
			return [kernel_size]
		else:
			return list(kernel_size)

	@staticmethod
	def _validate_hidden_dim(hidden_dim):
		if hidden_dim is None:
			return []
		elif not hasattr(hidden_dim, "__iter__"):
			return [hidden_dim]
		else:
			return list(hidden_dim)

	@staticmethod
	def _validate_transfer(transfer):
		if isinstance(transfer, str) or not hasattr(transfer, "__iter__"):
			return [transfer]
		else:
			return list(transfer)

	@staticmethod
	def _get_model(in_channels, out_channels, kernel_sizes, dims, transfer):
		layers = list()
		for in_channel,out_channel,kernel_size in zip(in_channels, out_channels, kernel_sizes):
			layers.append( torch.nn.Conv1d(in_channel,out_channel,kernel_size) )
		layers.append( View(-1,dims[0]) )
		for in_dim,out_dim,transfer in zip(dims[:-1], dims[1:], transfer):
			layers.append( torch.nn.Linear(in_dim,out_dim) )
			if transfer is not None:
				layers.append( _CNN._get_transfer(transfer) )
		model = torch.nn.Sequential(*layers)
		return model

	@staticmethod
	def _get_transfer(transfer):
		if transfer is None:
			return torch.nn.Identity()
		elif transfer == "leaky_relu":
			return torch.nn.LeakyReLU(inplace=True)
		elif transfer == "relu":
			return torch.nn.ReLU(inplace=True)
		elif transfer == "sigmoid":
			return torch.nn.Sigmoid()
		elif transfer == "tanh":
			return torch.nn.Tanh()
		else:
			raise ValueError(f"unexpected transfer function: {transfer}")

class CNN(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def __init__(
		self,
		window=100,
		horizon=100,
		n_filters=1,
		kernel_size=2,
		hidden_dim=100,
		transfer="relu",
		n_epochs=200,
		batch_learning=True,
		batch_size=128,
		shuffle=True,
		optimizer=torch.optim.Adam,
		optimizer_kwargs={},
		loss_function=torch.nn.MSELoss,
		loss_function_kwargs={},
		compute_device="cpu",
		parallel=False,
		early_stopping=False,
		validation_fraction=0.1,
		verbose=0,
		verbose_epoch_mod=10,
		verbose_batch_mod=10,
	):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)

		self.to(compute_device)

	def fit(self, X, y=None, **kwargs):
		# Build the model
		self._model = self._train_simple_classifier(X, y, **kwargs)

		# Return the classifier
		return self

	def transform(self, X):
		# Check if fit had been called
		sklearn.utils.validation.check_is_fitted(self, ['_model'])

		padding = np.full(
			shape=(self.window+self.horizon-1,)+self._model.output_shape,
			fill_value=np.nan
		)

		results = list()
		with torch.no_grad():
			self._model.eval()
			for X_seq in X:
				sequence_dataset = SubsequenceDataset(
					np.asarray(X_seq, dtype=np.float32),
					None,
					self.window,
					self.horizon
				)
				tmp_dataloader = torch.utils.data.DataLoader(
					sequence_dataset,
					batch_size=int(np.ceil(len(sequence_dataset)/5)),
					shuffle=False
				)
				outputs = list()
				for X_tmp in tmp_dataloader:
					X_tmp = X_tmp.to(self.device)
					tmp_outputs = self._model(X_tmp)
					outputs.append(tmp_outputs.detach().cpu().numpy())
				outputs = np.concatenate(outputs)
				results.append( 
					np.concatenate([padding, outputs])
				)
		return results

	def predict(self, X):
		return self.transform(X)

	def to(self, *args, **kwargs):
		self.device = torch.device(*args, **kwargs)
		try:
			sklearn.utils.validation.check_is_fitted(self, ['_model'])
		except:
			pass
		else:
			self._model = self._model.to(self.device)
		return self

	def _train_simple_classifier(self, X, y, filter_function=None):
		sequence_datasets = [
			SubsequenceDataset(
				np.asarray(X_tmp, dtype=np.float32),
				np.asarray(y_tmp, dtype=np.float32),
				self.window,
				self.horizon
			)
			for X_tmp, y_tmp in zip(X,y)
		]
		dataset = torch.utils.data.ConcatDataset(sequence_datasets)

		X_sample, y_sample = dataset[0]
		input_shape = X_sample.shape
		output_shape = y_sample.shape
		model = _CNN(
			input_shape=input_shape,
			output_shape=output_shape,
			n_filters=self.n_filters,
			kernel_size=self.kernel_size,
			hidden_dim=self.hidden_dim,
			transfer=self.transfer
		).to(self.device)

		if self.parallel:
			model = torch.nn.DataParallel(model)

		loss_function = self.loss_function(**self.loss_function_kwargs)

		optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

		model, self.train_loss, self.validate_loss = train_model(
			model,
			dataset,
			loss_function,
			optimizer,
			n_epochs=self.n_epochs,
			batch_size=self.batch_size,
			batch_learning=self.batch_learning,
			shuffle=self.shuffle,
			device=self.device,
			validation_fraction=self.validation_fraction,
			verbose=self.verbose,
			verbose_batch_mod=self.verbose_batch_mod,
			verbose_epoch_mod=self.verbose_epoch_mod,
			sequence_datasets=sequence_datasets,
			filter_function=filter_function
		)
		return model

