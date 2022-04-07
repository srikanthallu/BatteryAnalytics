import numpy as np
import sklearn.model_selection

import torch
import torch.nn
import torch.optim
import torch.utils.data

from .datasets import DatasetReindexer
from ..utils import time_function
	
def _calculate_loss(
	model,
	dataloader,
	loss_function,
	device
):
	with torch.no_grad():
		model.eval()
		total_loss = 0.
		for batch, (X_loc_train, y_loc_train) in enumerate(dataloader, 1):
			X_loc_train = X_loc_train.to(device)
			y_loc_train = y_loc_train.to(device)
			outputs = model(X_loc_train)
			loss = loss_function(outputs, y_loc_train)
			total_loss += loss.item() if loss_function.reduction=="sum" else loss.item()*len(X_loc_train)
		loss = total_loss if loss_function.reduction=="sum" else total_loss/len(dataloader.dataset)
		return loss

def _display_minibatch_progress(
	epoch,
	n_epochs,
	batch,
	total_samples,
	loss,
	train_dataloader,
	verbose,
	verbose_epoch_mod,
	verbose_batch_mod
):
	if verbose > 0:
		n_epochs_digits = int(np.log10(n_epochs)) + 1
		n_dataset_digits = int(np.log10(len(train_dataloader.dataset))) + 1
		n_batch_digits = int(np.log10(len(train_dataloader))) + 1
		if (
			(
				(
					(
						(verbose == 3) and 
						(epoch == n_epochs or epoch % verbose_epoch_mod == 0)
					) or
					(verbose == 5)
				) and
				((batch % verbose_batch_mod == 0) or (batch == len(train_dataloader)))
			) or
			(verbose >= 6)
		):
			message = "Train Epoch: {0} [{1}/{2} ({3}/{4}) {5:6.2f}%]   Loss: {6:15.6f}".format(
				str(epoch).rjust(n_epochs_digits, " "),
				str(total_samples).rjust(n_dataset_digits, " "),
				len(train_dataloader.dataset),
				str(batch).rjust(n_batch_digits, " "),
				len(train_dataloader),
				100. * batch / len(train_dataloader),
				loss.item()
			)
			print(message)

def _display_epoch_progress(
	model,
	epoch,
	n_epochs,
	loss_function,
	train_dataloader,
	validate_dataloader,
	device,
	train_loss,
	validate_loss,
	verbose,
	verbose_epoch_mod
):
	if verbose > 0:
		loss = _calculate_loss(model, train_dataloader, loss_function, device)
		train_loss.append( loss )
		if validate_dataloader is not None:
			validate_loss.append( _calculate_loss(model, validate_dataloader, loss_function, device) )
		n_epochs_digits = int(np.log10(n_epochs)) + 1
		n_dataset_digits = int(np.log10(len(train_dataloader.dataset))) + 1
		n_batch_digits = int(np.log10(len(train_dataloader))) + 1
		if (
			(epoch == n_epochs) or
			((verbose == 2 or verbose == 3) and epoch % verbose_epoch_mod == 0) or
			(verbose >= 4)
		):
			message = "Train Epoch: {}, Loss: {:15.6f}".format(
				str(epoch).rjust(n_epochs_digits, " "),
				loss
			)
			print(message)
		if (
			((verbose == 3) and (epoch == n_epochs or epoch % verbose_epoch_mod == 0)) or
			(verbose >= 5)
		):
			print()


def batch_train(
	model,
	train_dataloader,
	validate_dataloader,
	loss_function,
	optimizer,
	n_epochs,
	batch_size,
	device,
	verbose,
	verbose_epoch_mod,
	sequence_datasets,
	filter_function,
):
	train_loss = list()
	validate_loss = list()
	_display_epoch_progress(model, 0, n_epochs, loss_function, train_dataloader, validate_dataloader, device, train_loss, validate_loss, verbose, verbose_epoch_mod)
	for epoch in range(1, n_epochs+1):
		total_samples = 0
		model.train()
		loss = list()

		# clear any calculated gradients
		optimizer.zero_grad()

		for batch, (X_loc_train, y_loc_train) in enumerate(train_dataloader, 1):
			total_samples += len(X_loc_train)
			
			# ensure the data lives on the correct device
			X_loc_train = X_loc_train.to(device)
			y_loc_train = y_loc_train.to(device)

			# forward pass, compute outputs
			outputs = model(X_loc_train)

			# compute loss
			loss.append( loss_function(outputs, y_loc_train) )
		loss = sum(loss) / total_samples

		# backward pass, compute gradients
		loss.backward()

		# update learnable parameters
		optimizer.step()

		if filter_function is not None and epoch > n_epochs/3:
			model.train()
			for ii,sequence_dataset in enumerate(sequence_datasets):
				tmp_dataloader = torch.utils.data.DataLoader(
					sequence_dataset,
					batch_size=int(np.ceil(len(sequence_dataset)/5)),
					shuffle=False
				)
				with torch.no_grad():
					targets = list()
					for X_tmp,y_tmp in tmp_dataloader:
						X_tmp = X_tmp.to(device)
						tmp_outputs = model(X_tmp)
						targets.append(tmp_outputs.detach().cpu().numpy())
					targets = np.concatenate(targets).astype(np.float64)
					targets = torch.from_numpy( 
						filter_function(targets.transpose()).transpose().astype(np.float32)
					).to(device)
				offset = 0
				optimizer.zero_grad()
				for X_tmp,y_tmp in tmp_dataloader:
					X_tmp = X_tmp.to(device)
					tmp_outputs = model(X_tmp)
					loss = loss_function(tmp_outputs, targets[offset:offset+len(tmp_outputs)])
					loss.backward()
					offset += len(tmp_outputs)
				optimizer.step()
		
		_display_epoch_progress(model, epoch, n_epochs, loss_function, train_dataloader, validate_dataloader, device, train_loss, validate_loss, verbose, verbose_epoch_mod)
	train_loss = np.asarray( train_loss )
	validate_loss = np.asarray( validate_loss )
	return model, train_loss, validate_loss

def minibatch_train(
	model,
	train_dataloader,
	validate_dataloader,
	loss_function,
	optimizer,
	n_epochs,
	batch_size,
	device,
	verbose,
	verbose_batch_mod,
	verbose_epoch_mod,
	sequence_datasets,
	filter_function,
):
	train_loss = list()
	validate_loss = list()
	_display_epoch_progress(model, 0, n_epochs, loss_function, train_dataloader, validate_dataloader, device, train_loss, validate_loss, verbose, verbose_epoch_mod)
	for epoch in range(1, n_epochs+1):
		total_samples = 0

		for batch, (X_loc_train, y_loc_train) in enumerate(train_dataloader, 1):
			total_samples += len(X_loc_train)
			model.train()

			# clear any calculated gradients
			optimizer.zero_grad()

			# ensure the data lives on the correct device
			X_loc_train = X_loc_train.to(device)
			y_loc_train = y_loc_train.to(device)

			# forward pass, compute outputs
			outputs = model(X_loc_train)

			# compute loss
			loss = loss_function(outputs, y_loc_train)
			
			# backward pass, compute gradients
			loss.backward()
		
			# update learnable parameters
			optimizer.step()

			_display_minibatch_progress(epoch, n_epochs, batch, total_samples, loss, train_dataloader, verbose, verbose_epoch_mod, verbose_batch_mod)

		if filter_function is not None and epoch > n_epochs/3:
			model.train()
			for ii,sequence_dataset in enumerate(sequence_datasets):
				tmp_dataloader = torch.utils.data.DataLoader(
					sequence_dataset,
					batch_size=int(np.ceil(len(sequence_dataset)/5)),
					shuffle=False
				)
				with torch.no_grad():
					targets = list()
					for X_tmp,y_tmp in tmp_dataloader:
						X_tmp = X_tmp.to(device)
						tmp_outputs = model(X_tmp)
						targets.append(tmp_outputs.detach().cpu().numpy())
					targets = np.concatenate(targets).astype(np.float64)
					targets = torch.from_numpy( 
						filter_function(targets.transpose()).transpose().astype(np.float32)
					).to(device)
				offset = 0
				optimizer.zero_grad()
				for X_tmp,y_tmp in tmp_dataloader:
					X_tmp = X_tmp.to(device)
					tmp_outputs = model(X_tmp)
					loss = loss_function(tmp_outputs, targets[offset:offset+len(tmp_outputs)])
					loss.backward()
					offset += len(tmp_outputs)
				optimizer.step()
		
		_display_epoch_progress(model, epoch, n_epochs, loss_function, train_dataloader, validate_dataloader, device, train_loss, validate_loss, verbose, verbose_epoch_mod)
	train_loss = np.asarray( train_loss )
	validate_loss = np.asarray( validate_loss )
	return model, train_loss, validate_loss

@time_function("train_model")
def train_model(
	model,
	dataset,
	loss_function,
	optimizer,
	n_epochs=100,
	batch_size=32,
	batch_learning=False,
	shuffle=True,
	device="cpu",
	validation_fraction=0.0,
	verbose=0,
	verbose_batch_mod=10,
	verbose_epoch_mod=10,
	sequence_datasets=None,
	filter_function=None,
):
	if validation_fraction > 0:
		train_idx, validate_idx = sklearn.model_selection.train_test_split(
			range(len(dataset))
		)
		
		train_dataset = DatasetReindexer(dataset, train_idx)
		train_dataloader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)

		validate_dataset = DatasetReindexer(dataset, validate_idx)
		validate_dataloader = torch.utils.data.DataLoader(
			validate_dataset,
			batch_size=batch_size,
			shuffle=False
		)
	else:
		train_dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)
		validate_dataloader = None

	if batch_learning:
		return batch_train(
			model,
			train_dataloader,
			validate_dataloader,
			loss_function,
			optimizer,
			n_epochs,
			batch_size,
			device,
			verbose,
			verbose_epoch_mod,
			sequence_datasets,
			filter_function
		)
	else:
		return minibatch_train(
			model,
			train_dataloader,
			validate_dataloader,
			loss_function,
			optimizer,
			n_epochs,
			batch_size,
			device,
			verbose,
			verbose_batch_mod,
			verbose_epoch_mod,
			sequence_datasets,
			filter_function
		)

