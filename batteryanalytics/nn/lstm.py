import inspect

import numpy as np
import sklearn
import sklearn.base

import torch
import torch.nn
import torch.optim
import torch.utils.data

from .datasets import SubsequenceDataset
from .train import train_model

class _LSTM(torch.nn.Module):
	def __init__(self, input_shape, output_shape, hidden_dim):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)
		self._lstm = torch.nn.LSTM(self.input_shape[1], hidden_dim, batch_first=True)
		self._linear = torch.nn.Linear(hidden_dim, self.output_shape[-1])

	def forward(self, X, hidden_state=None):
		n_samples,seq_len,n_features = X.shape
		output, hidden_state = self._lstm(X, hidden_state)
		if len(self.output_shape) == 1:
			output = output[:,-1,:]
		output = output.reshape(-1, self.hidden_dim)
		output = self._linear( output )
		if len(self.output_shape) != 1:
			output = output.reshape(n_samples, seq_len, self.output_shape[-1])
		return output

class LSTM(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def __init__(
		self,
		window=100,
		horizon=100,
		hidden_dim=100,
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
	
	def predict(self, X, hidden_state=None):
		return self.transform(X, hidden_state)

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
		model = _LSTM(
			input_shape=input_shape,
			output_shape=output_shape,
			hidden_dim=self.hidden_dim,
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

class GlobalLSTM(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	def __init__(
		self,
		window=100,
		horizon=100,
		hidden_dim=100,
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
	
	def predict(self, X, hidden_state=None):
		return self.transform(X, hidden_state)

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
		print(type(X))
		print(type(y))

		print(len(X), type(X[0]))

		dataset = torch.utils.data.TensorDataset(
			torch.from_numpy(np.asarray(X, dtype=np.float32)),
			torch.from_numpy(np.asarray(y, dtype=np.float32))
		)
		sequence_datasets = None

		X_sample, y_sample = dataset[0]
		input_shape = X_sample.shape
		output_shape = y_sample.shape
		model = _LSTM(
			input_shape=input_shape,
			output_shape=output_shape,
			hidden_dim=self.hidden_dim,
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


