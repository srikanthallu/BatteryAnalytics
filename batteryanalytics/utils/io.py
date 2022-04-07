import errno
import joblib
import os
import re
from collections import namedtuple

import numpy as np

from .utils import time_function

def mkdirs(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def sanitize_string(string):
	string = re.sub(r"[^\w\s-]", "", string.lower())
	return re.sub(r"[\s]+", "_", string).strip("-_")

def sanitize_strings(strings):
	if isinstance(strings, str):
		strings = [strings]
	return "-".join(map(
		sanitize_string,
		strings
	))

def sanitize_int(num):
	return f"{num:03d}"

def sanitize_ints(nums):
	if not hasattr(nums, "__iter__"):
		nums = [nums]
	return "-".join(map(
		sanitize_int,
		nums
	))

def sanitize_feature_name(feature_name):
	return sanitize_string(feature_name)

def sanitize_feature_names(feature_names):
	return sanitize_strings(feature_names)

def sanitize_transfer_names(transfer_names):
	return sanitize_strings(transfer_names)

def sanitize_hidden_dims(hidden_dims):
	return sanitize_ints(hidden_dims)

def sanitize_n_filters(n_filters):
	return sanitize_ints(n_filters)

def sanitize_kernel_sizes(kernel_sizes):
	return sanitize_ints(kernel_sizes)

def sanitize_holdout_name(holdout):
	return "-".join("_".join(item.split()) for item in holdout)

def get_ann_dirname(holdouts):
	return "./ann_models/{}".format(
		"/".join(map(
			sanitize_holdout_name,
			sorted(holdouts)
		)
	))

def get_ann_basename(features, window, horizon, hidden_dims, transfer, n_epochs, training_filter, model_id):
	features = sanitize_feature_names(features)
	hidden_dims = sanitize_hidden_dims(hidden_dims)
	transfer = sanitize_transfer_names(transfer)
	return (
		f"ann.{features}"
		f".{window:03d}-window."
		f"{horizon:03d}-horizon."
		f"{hidden_dims}-dim."
		f"{transfer}-transfer."
		f"{n_epochs:03d}-epochs."
		f"{training_filter}-filter."
		f"{model_id}-id.joblib"
	)

def get_cnn_dirname(holdouts):
	return "./cnn_models/{}".format(
		"/".join(map(
			sanitize_holdout_name,
			sorted(holdouts)
		)
	))

def get_cnn_basename(features, window, horizon, n_filters, kernel_sizes, hidden_dims, transfer, n_epochs, training_filter, model_id):
	features = sanitize_feature_names(features)
	n_filters = sanitize_n_filters(n_filters)
	kernel_sizes = sanitize_kernel_sizes(kernel_sizes)
	hidden_dims = sanitize_hidden_dims(hidden_dims)
	transfer = sanitize_transfer_names(transfer)
	return (
		f"cnn.{features}"
		f".{window:03d}-window."
		f"{horizon:03d}-horizon."
		f"{n_filters}-filters."
		f"{kernel_sizes}-kernels."
		f"{hidden_dims}-dim."
		f"{transfer}-transfer."
		f"{n_epochs:03d}-epochs."
		f"{training_filter}-filter."
		f"{model_id}-id.joblib"
	)

def get_lstm_dirname(holdouts):
	return "./lstm_models/{}".format(
		"/".join(map(
			sanitize_holdout_name,
			sorted(holdouts)
		)
	))

def get_lstm_basename(features, window, horizon, hidden_dim, n_epochs, training_filter, model_id):
	features = sanitize_feature_names(features)
	hidden_dim = sanitize_hidden_dims(hidden_dim)
	return (
		f"lstm.{features}"
		f".{window:03d}-window."
		f"{horizon:03d}-horizon."
		f"{hidden_dim}-dim."
		f"{n_epochs:03d}-epochs."
		f"{training_filter}-filter."
		f"{model_id}-id.joblib"
	)

def save_model(filename, model):
	mkdirs(os.path.dirname(filename))
	model.to("cpu")
	if model.parallel:
		model.parallel = False
		model._model = model._model.module
	joblib.dump(model, filename)
	return filename

ANNModelKey = namedtuple("ANNModelKey", "holdouts variables window horizon dim transfer training_filter n_epochs model_id")

def load_ann_models(directory, variable_mapping):
	if not directory.endswith(".joblib"):
		directory += "/"
	
	models = dict()
	predictions = dict()
	for dirpath, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			if filename.endswith(".joblib") or filename.endswith(".npz"):
				holdouts = tuple(sorted(map(
					lambda x: tuple(item.replace("_", " ") for item in x.split("-")),
					dirpath.replace(directory, "").split("/")
				)))
				tokens = filename.split(".")
				variables = tuple(tokens[1].split("-"))
				window = int(tokens[2].split("-")[0])
				horizon = int(tokens[3].split("-")[0])
				dim = tuple(map(int, tokens[4].split("-")[:-1]))
				transfer = tuple(tokens[5].split("-")[:-1])
				n_epochs = int(tokens[6].split("-")[0])
				training_filter = tokens[7].split("-")[0]
				model_id = int(tokens[8].split("-")[0])
				
#				print("\n".join(map(str, zip(
#					range(len(tokens)),
#					tokens
#				))))
#				print(f"holdouts: {holdouts}")
#				print(f"variables: {variables}")
#				print(f"window: {window}")
#				print(f"horizon: {horizon}")
#				print(f"dim: {dim}")
#				print(f"transfer: {transfer}")
#				print(f"n_epochs: {n_epochs}")
#				print(f"training_filter: {training_filter}")
#				print(f"model_id: {model_id}")
#				print()

				variables = tuple(map(variable_mapping.get, variables))
				model_key = ANNModelKey(holdouts, variables, window, horizon, dim, transfer, training_filter, n_epochs, model_id)
				if filename.endswith(".joblib"):
					models[model_key] = joblib.load(os.path.join(dirpath,filename))
				else:
					with np.load(os.path.join(dirpath,filename)) as npz_file:
						predictions[model_key] = {
							tuple(item.replace("_", " ") for item in key.split("-")):np.squeeze(value)
							for key, value in npz_file.items()
						}
	return models, predictions

CNNModelKey = namedtuple("CNNModelKey", "holdouts variables window horizon filters kernels dim transfer n_epochs training_filter model_id")

def load_cnn_models(directory, variable_mapping):
	if not directory.endswith(".joblib"):
		directory += "/" 

	models = dict()
	predictions = dict()
	for dirpath, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			if filename.endswith(".joblib") or filename.endswith(".npz"):
				holdouts = tuple(sorted(map(
					lambda x: tuple(item.replace("_", " ") for item in x.split("-")),
					dirpath.replace(directory, "").split("/")
				))) 
				tokens = filename.split(".")
				variables = tuple(tokens[1].split("-"))
				window = int(tokens[2].split("-")[0])
				horizon = int(tokens[3].split("-")[0])
				filters = tuple(map(int, tokens[4].split("-")[:-1]))
				kernels = tuple(map(int, tokens[5].split("-")[:-1]))
				dim = int(tokens[6].split("-")[0])
				transfer = tuple(tokens[7].split("-"))
				n_epochs = int(tokens[8].split("-")[0])
				training_filter = tokens[9].split("-")[0]
				model_id = int(tokens[10].split("-")[0])

#				print("\n".join(map(str, zip(
#					range(len(tokens)),
#					tokens
#				))))
#				print(f"holdouts: {holdouts}")
#				print(f"variables: {variables}")
#				print(f"window: {window}")
#				print(f"horizon: {horizon}")
#				print(f"filters: {filters}")
#				print(f"kernels: {kernels}")
#				print(f"dim: {dim}")
#				print(f"transfer: {transfer}")
#				print(f"n_epochs: {n_epochs}")
#				print(f"training_filter: {training_filter}")
#				print(f"model_id: {model_id}")
#				print()
				
				variables = tuple(map(variable_mapping.get, variables))
				model_key = CNNModelKey(holdouts, variables, window, horizon, filters, kernels, dim, transfer, n_epochs, training_filter, model_id)
				if filename.endswith(".joblib"):
					models[model_key] = joblib.load(os.path.join(dirpath,filename))
				else:
					with np.load(os.path.join(dirpath,filename)) as npz_file:
						predictions[model_key] = { 
							tuple(item.replace("_", " ") for item in key.split("-")):np.squeeze(value)
							for key, value in npz_file.items()
						}   
	return models, predictions

LSTMModelKey = namedtuple("LSTMModelKey", "holdouts variables window horizon dim training_filter n_epochs model_id")
def load_lstm_models(directory, variable_mapping):
	if not directory.endswith(".joblib"):
		directory += "/"

	models = dict()
	predictions = dict()
	for dirpath, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			if filename.endswith(".joblib") or filename.endswith(".npz"):
				holdouts = tuple(sorted(map(
					lambda x: tuple(item.replace("_", " ") for item in x.split("-")),
					dirpath.replace(directory, "").split("/")
				)))
				tokens = filename.split(".")
				variables = tuple(tokens[1].split("-"))
				window = int(tokens[2].split("-")[0])
				horizon = int(tokens[3].split("-")[0])
				dim = int(tokens[4].split("-")[0])
				n_epochs = int(tokens[5].split("-")[0])
				training_filter = tokens[6].split("-")[0]
				model_id = int(tokens[7].split("-")[0])

#				print("\n".join(map(str, zip(
#					range(len(tokens)),
#					tokens
#				))))
#				print(f"holdouts: {holdouts}")
#				print(f"variables: {variables}")
#				print(f"window: {window}")
#				print(f"horizon: {horizon}")
#				print(f"dim: {dim}")
#				print(f"n_epochs: {n_epochs}")
#				print(f"training_filter: {training_filter}")
#				print(f"model_id: {model_id}")
#				print()

				variables = tuple(map(variable_mapping.get, variables))
				model_key = LSTMModelKey(holdouts, variables, window, horizon, dim, training_filter, n_epochs, model_id)
				if filename.endswith(".joblib"):
					models[model_key] = joblib.load(os.path.join(dirpath,filename))
				else:
					with np.load(os.path.join(dirpath,filename)) as npz_file:
						predictions[model_key] = {
							tuple(item.replace("_", " ") for item in key.split("-")):np.squeeze(value)
							for key, value in npz_file.items()
						}
	return models, predictions
