""" TransmissionForestVis.py

Use trajectories and distributions from IndividualLevelDistributions.py to sample
and visualize a plausible transmission forest"""
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6)

## For sampling, etc.
from methods.seir import LogNormalSEIR, sample_traj

## For working with graphs
import networkx as nx

## General environment setup stuff
plt.rcParams["font.size"] = 22

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def random_rewiring_sampler(N,I,ks_given_N,non_zero_p_k,
							num_samples=10000,verbose=False,shuffle=True):
	x = np.zeros((num_samples,I))
	x[:,:N] = 1.
	for i in range(num_samples):
		n = 0
		this_sample = x[i].copy()
		while n < N:
			options = np.sum(this_sample[n:]).astype(int)
			rw = np.random.choice(ks_given_N[1:options+1],
								  p=non_zero_p_k[:options]/(1.-np.sum(non_zero_p_k[options:])))
			this_sample[n] = rw
			this_sample[n+1:n+rw] = 0
			n = n + rw
			if verbose:
				print("node = {}".format(n))
				print("draw = {}".format(rw))
				print(this_sample)
		if shuffle:
			np.random.shuffle(this_sample)
		x[i] = this_sample
	return x

if __name__ == "__main__":

	## Get the model outputs from prevalence.py
	fit_df = pd.read_pickle("pickle_jar\\state_model.pkl")

	## Set up time axis related parameters
	ticks = pd.date_range("2020-03-01",fit_df.index[-1],freq="MS")
	tick_labels = [t.strftime("%b")+(t.month == 1)*"\n{}".format(t.year) for t in ticks]

	## Get the population information
	pyramid = pd.read_csv("_data\\age_pyramid.csv")
	pyramid = pyramid.set_index("age_bin")["population"]
	population = pyramid.sum()

	## Create a model
	model = LogNormalSEIR(S0=population,
						  D_e=5,
						  D_i=4,
						  z_t=fit_df["z_t"].values)

	## Sample some trajectories
	num_samples = 10000
	ic = np.array([model.S0*np.ones((num_samples,)),
				   np.zeros((num_samples,)),
				   np.random.normal(fit_df["pulse"].values[0],
				   					fit_df["pulse_std"].values[0],
				   					size=(num_samples,))]).T
	ic = np.clip(ic,0,None)
	samples = sample_traj(model,
						  fit_df["beta_t"].values,
						  sig_eps=fit_df["sig_eps"].values,
						  z_t=fit_df["z_t"].values,
						  avg_v_t=fit_df["vax_avg"].values,
						  cov_v_t=np.diag(fit_df["vax_var"].values),
						  ic=ic,
						  )

	## Collect the relevant trajectories - new exposures every day and the
	## infectious population responsible.
	new_exposures = samples[:,1,1:] - (1. - (1./model.D_e))*samples[:,1,:-1]
	new_exposures = pd.DataFrame(new_exposures.T,
								 index=fit_df.index[:-1]).reindex(fit_df.index).fillna(0).T
	infectious_pop = pd.DataFrame(samples[:,2,:].T,
								index=fit_df.index).T

	## Pick a particular sample
	N_t = np.round(new_exposures.loc[908]).astype(np.int32)

	## Compute the (approximate) associated infections
	I_t = np.convolve(N_t.values,
					  np.array(model.D_i*[1]+model.D_e*[0]+[0]), 
					  mode="same")
	I_t = pd.Series(I_t,index=N_t.index)
	I_t = I_t.shift(model.D_e+model.D_i).fillna(0).astype(np.int32)
	
	## To avoid edge effects, and the importations,
	## slice to mid February at least.
	N_t = N_t.loc["2020-02-25":"2020-09-01"]
	I_t = I_t.loc["2020-02-25":"2020-09-01"]

	## Get the daily degree distributions and align it
	## to the N_t series.
	pk_df = pd.read_pickle("pickle_jar\\pk_timeseries.pkl")
	pk_df = pk_df.reindex(N_t.index).fillna(method="bfill")
	max_event_size = max(pk_df.columns[-1],N_t.max()+1)
	pk_df = pk_df.T.reindex(np.arange(0,max_event_size,dtype=np.int32)).fillna(0).T

	## Create a dataframe that organizes the node attributes.
	df = pd.DataFrame(np.arange(N_t.sum(),dtype=np.int32)[:,np.newaxis],
					  columns=["node"])
	df["time"] = np.repeat(N_t.index,N_t.values)
	df["inf_start"] = df["time"] + pd.to_timedelta(model.D_e,unit="d")
	df["inf_end"] = df["inf_start"] + pd.to_timedelta(model.D_i,unit="d")
	df["parent"] = len(df)*[None]
	
	## Loop over days, with a set random seed
	## for reproducibility in other scripts.
	np.random.seed(23)
	for d in N_t.index[model.D_e+model.D_i+1:]:

		## Get the information necessary to sample this days
		## bipartite graph. Start by refining the distributions given N.
		N = N_t.loc[d]
		I = I_t.loc[d]
		p_k = pk_df.loc[d]
		p_k_given_N = p_k.loc[:N]/(1. - (p_k.loc[N+1:].sum()))
		non_zero_p_k = p_k_given_N.values[1:]/(1.-p_k_given_N.loc[0])

		## Sample a bipartite graph
		sample = random_rewiring_sampler(N,I,
										 p_k_given_N.index,
										 non_zero_p_k,
										 num_samples=1,
										 )[0]
		
		## Get the subsets of the nodes associated with the graph
		inf_df = df.loc[(df["inf_start"] <= d) & (df["inf_end"] > d)]
		exp_df = df.loc[df["time"] == d]
		
		## And then compute parents associated with the graph
		df.loc[exp_df.index,"parent"] = np.repeat(inf_df.index,sample)
	print("\nThis tree sample:")
	print(df)

	## Compute a statistic...
	nodes = set(df["node"])
	parents = set(df["parent"])
	child_less = nodes.difference(parents)
	print("Childless nodes = {} ({}%)".format(len(child_less),
											  100*len(child_less)/len(nodes)))

	## Use the dataframe to make a graph
	G = nx.DiGraph()
	for t, sf in df.groupby("time"):
		G.add_nodes_from(sf["node"],
						 time=t)
	edges = df.loc[df["parent"].notnull(),["parent","node"]]
	G.add_edges_from(edges.values)
	undirected_G = G.to_undirected()

	## Get a particular, big chain, to see.
	x = df["parent"].value_counts()
	ss_node = x.index[41]
	ex_chain = G.subgraph(nx.dfs_preorder_nodes(G,ss_node))
	undirected_chain = ex_chain.to_undirected()
	ssG = G.subgraph([ss_node])
	print("Length of the chain example = {} nodes".format(len(ex_chain)))
	
	## Compute x-y positions (are there smarter ways to do this?)
	## First as a bar chart, then as a clock.
	bar_pos = {}
	for t, sf in df.groupby("time"):
		d = (t - N_t.index[0]).days
		this_day = {n:(d,i) for i,n in enumerate(sf["node"])}
		bar_pos.update(this_day)

	## Make a figure for the vis (fig 5b)
	fig, tree_ax = plt.subplots(figsize=(12,6))

	## Plot the bar graph version
	axes_setup(tree_ax)
	nx.draw_networkx_edges(undirected_G,
						   bar_pos, 
						   ax=tree_ax,
						   alpha=0.0025,
						   edge_color="grey",
						   )
	nx.draw_networkx_nodes(ex_chain,
						   bar_pos, 
						   ax=tree_ax,
						   node_size=10,
						   node_color="k",
						   )
	nx.draw_networkx_nodes(ssG,
						   bar_pos, 
						   ax=tree_ax,
						   node_size=90,
						   node_color="C3",
						   )
	nx.draw_networkx_edges(undirected_chain,
						   bar_pos, 
						   ax=tree_ax,
						   alpha=0.9,
						   edge_color="xkcd:saffron",
						   )
	
	## Overlay the time series
	tree_ax.plot(N_t.values,color="k",lw=2,zorder=0)
	
	## Set up the ticks
	tree_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
	tree_ax.set_ylabel("Daily Covid exposures")
	tree_ax.set_ylim((0,None))
	ticks = pd.date_range(N_t.index[0],N_t.index[-1],freq="MS")
	tree_ax.set_xticks([(d-N_t.index[0]).days for d in ticks])
	tree_ax.set_xticklabels([t.strftime("%b")+(t.month == 1)*"\n{}".format(t.year) for t in ticks])
	tree_ax.set_xlim((0,(pd.to_datetime("2020-06-01")-N_t.index[0]).days))
	tree_ax.set_xlabel("Month in the first wave")
	
	## Adjust it
	fig.tight_layout()
	fig.savefig("_plots\\transmission_tree_vis.png")
	
	## Done
	plt.show()