import inspect
import numpy as np

import torch
import torch.utils.data

class SubsequenceDataset(torch.utils.data.Dataset):
	def __init__(self, X, y, window, horizon):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)
		if self.y is not None:
			assert len(self.X) == len(self.y), "size mismatch between X and y"

	def __len__(self):
		return len(self.X) - self.window - self.horizon + 1

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			return self.__getslice(idx)
		else:
			return self.__getitem(idx)
			
	def __getitem(self, idx):
		if idx < 0:
			idx = len(self) + idx
		if idx >= len(self):
			raise IndexError("list index out of range")
		X_ret = self.X[idx : idx+self.window]
		if self.y is None:
			return X_ret
		y_ret = self.y[idx + self.window + self.horizon - 1]
		return X_ret, y_ret
			
	def __getslice(self, idx):
		return np.asarray([
			self.__getitem__(ii)
			for ii in range(*idx.indices(len(self)))
		])

class DatasetReindexer(torch.utils.data.Dataset):
	def __init__(self, dataset, idx1):
		super().__init__()
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")
		for arg, val in values.items():
			setattr(self, arg, val)

	def __len__(self):
		return len(self.idx1)

	def __getitem__(self, idx2):
		return self.dataset[ self.idx1[idx2] ]
