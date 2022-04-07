import itertools
import os
import sys

import numpy as np
import pandas as pd
import scipy.signal
import sklearn.isotonic
import sklearn.model_selection

try:
	import batteryanalytics
except ModuleNotFoundError as ie:
	sys.path.append( os.path.join(os.path.abspath(""), "../") )
	import batteryanalytics
from batteryanalytics import utils as ba_utils
from batteryanalytics.nn import LSTM

@ba_utils.time_function("load_data:")
def load_data(filename):
	raw_df = pd.read_csv(filename, header=[0,1,2], index_col=0, compression="gzip")
	return raw_df

@ba_utils.time_function("preprocess_data:")
def preprocess_data(raw_df, features, target):
	data_df = raw_df.copy()

	levels_0 = list()
	levels_1 = list()
	for column in data_df.columns:
		if column[0] not in levels_0:
			levels_0.append(column[0])
		if column[1] not in levels_1:
			levels_1.append(column[1])

	samples = {
		pair:data_df.xs(pair, axis="columns", level=(0,1), drop_level=False).dropna()
		for pair in itertools.product(levels_0, levels_1)
	}
	for sample_df in samples.values():
		series = sample_df.loc[:,pd.IndexSlice[:,:,target]].copy()
		idx = np.squeeze(series.values).argmax()
		sample_df.drop(sample_df.index[2*idx+1:], inplace=True)
		# vmin, vmax = series.min(), series.max()
		# delta = vmax - vmin
		# idx = series > 0.05*delta + vmin
		# cut_off = np.squeeze(idx.iloc[::-1].idxmax().values).item()
		# sample_df.drop(sample_df.index[sample_df.index > cut_off], inplace=True)

	return samples

@ba_utils.time_function("train_model:")
def train_model(samples, n_split, features, target, window, horizon, hidden_dim, n_epochs, training_filter, model_id):
	model = LSTM(
		window=window,
		horizon=horizon,
		hidden_dim=hidden_dim,
		n_epochs=n_epochs,
		batch_learning=False,
		batch_size=32,
		shuffle=True,
		compute_device="cpu",
		parallel=False,
		verbose=1,
	)

	samples_items = np.empty(shape=len(samples), dtype=object)
	samples_items[:] = list(samples.items())

	splitter = sklearn.model_selection.LeaveOneOut()
	#splitter = sklearn.model_selection.KFold(n_splits=10, random_state=0, shuffle=True)#LeaveOneOut()
	splits = list(splitter.split(samples_items))
	train_idx,test_idx = splits[n_split]

	XX = [
		value.loc[:,pd.IndexSlice[:,:,features]].values
		for key,value in samples_items[train_idx] 
	]
	yy = [
		value.loc[:,pd.IndexSlice[:,:,target]].values
		for key,value in samples_items[train_idx] 
	]
	if training_filter == "savgol":
		filter_function = lambda x:scipy.signal.savgol_filter(
			x,
			window_length=99,
			polyorder=2
		)
	elif training_filter == "isotonic_regression":
		filter_function = lambda x:np.vstack([
			np.hstack([
				sklearn.isotonic.isotonic_regression(row[:np.argmax(row)], increasing=True) if np.argmax(row) > 0 else [],
				sklearn.isotonic.isotonic_regression(row[np.argmax(row):], increasing=False) if np.argmax(row) < len(row) else []
			])
			for row in x
		])
	elif training_filter == "none":
		filter_function = None
	else:
		raise ValueError(f"Unexpected training_filter value ({filter_function})")

	model.fit(XX, yy, filter_function=filter_function)
	
	filename = os.path.join(
		ba_utils.get_lstm_dirname([key for key,value in samples_items[test_idx]]),
		ba_utils.get_lstm_basename(features, window, horizon, model.hidden_dim, model.n_epochs, training_filter, model_id)
	).replace("joblib", "npz")

	keys = list(map(ba_utils.sanitize_holdout_name, samples.keys()))
	values = list(map(
		np.squeeze,
		model.transform(
			list(map(
				lambda df:df.loc[:,pd.IndexSlice[:,:,features]].values,
				samples.values()
			))
		)
	))
	kwargs = dict(zip(keys,values))

	ba_utils.mkdirs(os.path.dirname(filename))
	np.savez(filename, **kwargs)
	
	filename = os.path.join(
		ba_utils.get_lstm_dirname([key for key,value in samples_items[test_idx]]),
		ba_utils.get_lstm_basename(features, model.window, model.horizon, model.hidden_dim, model.n_epochs, training_filter, model_id)
	)
	filename = ba_utils.save_model(filename, model)
	return model, filename

def main(infile, n_split, features, window, horizon, hidden_dim, n_epochs, training_filter, model_id):
	target = "Temperature [C]"
	
	raw_df = load_data(infile)
	print("{} Raw Data".format(raw_df.shape))
	samples = preprocess_data(raw_df, features, target)
	print("{} Samples".format(len(samples)))
	train_model(samples, n_split, features, target, window, horizon, hidden_dim, n_epochs, training_filter, model_id)

import argparse
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="FS LSTM",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	parser.add_argument(
		"--window",
		type=int,
		required=True,
		help="sample window length"
	)

	parser.add_argument(
		"--horizon",
		type=int,
		required=True,
		help="prediction horizon"
	)
	
	parser.add_argument(
		"--hidden-dim",
		type=int,
		required=True,
		help="LSTM hidden dimension size"
	)

	parser.add_argument(
		"--n-epochs",
		type=int,
		required=True,
		help="number of training epochs"
	)
	
	parser.add_argument(
		"--n-split",
		type=int,
		required=True,
		help="split position"
	)
	
	parser.add_argument(
		"--features",
		nargs="+",
		type=str,
		required=True,
		help="list of training features"
	)
	
	parser.add_argument(
		"--training-filter",
		type=str,
		required=True,
		help="filter to be applyed between training epochs"
	)
	
	parser.add_argument(
		"--model-id",
		type=int,
		required=True
	)
	
	parser.add_argument(
		"--infile",
		type=str,
		metavar="FILENAME",
		required=True,
		help="data input file"
	)
	
	args = vars(parser.parse_args())
	main(**args)
	
