import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def display_df_columns(df, figsize=(16,10), title=None, title_loc="center", xlabel=None, ylabel=None, log_scale=False, legend_kwargs=None, ax=None):
	ax_is_none = ax is None
	if ax_is_none:
		fig,ax = plt.subplots(tight_layout=True, figsize=figsize)
	for column in df.columns:
		ax.plot(df[column], label=column)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title, loc=title_loc)
	if log_scale:
		ax.set_yscale("log", base=10)
	if legend_kwargs is not None:
		ax.legend(**legend_kwargs)
	if ax_is_none:
		plt.show()

def display_bar_plot(df, yerr_df=None, figsize=(16,10), ax_title=None, label_rotation=None, label_alignment=None, log_base=None, ncol=5, filename=None):
	index = np.arange(len(df))
	width = 1/(len(df.columns)+1)
	fig,ax = plt.subplots(constrained_layout=True, figsize=figsize)
	for ii,column in enumerate(df.columns):
		if yerr_df is None:
			ax.bar(
				x=index + width*(ii-(len(df.columns)-1)/2),
				height=df[column],
				width=width,
				label=column
			)
		else:
			ax.bar(
				x=index + width*(ii-(len(df.columns)-1)/2),
				height=df[column],
				yerr=yerr_df[column],
				width=width,
				label=column
			)
	if log_base is not None:
		ax.set_yscale("log", base=log_base)
	ax.set_xticks(index)
	ax.set_xticklabels(df.index)
	for label in ax.get_xticklabels():
		if label_rotation is not None:
			label.set_rotation(label_rotation)
		if label_alignment is not None:
			label.set_ha(label_alignment)
	ax.set_title(ax_title)
	if len(df.columns) > 1:
		ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=ncol)
	# plt.tight_layout()
	if filename is not None:
		plt.savefig(filename)
	plt.show()

def display_loss_curve(model, **kwargs):
	df = pd.DataFrame()
	df["train"] = model.train_loss
	df["validate"] = model.validate_loss
	
	kwargs["df"] = df
	display_df_columns(**kwargs)

