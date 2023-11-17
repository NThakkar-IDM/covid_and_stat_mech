""" SampleTransmissionTrees.py

Script to create a dataset of sampled WA COVID-19 transmission trees
using the rewiring approach. These samples are used to validate the 
generating function based methods.

Note: Sampling is a little labor intensive, and this script takes a few
minutes to run. """
import sys
sys.path.append("..\\")

## Standard imports
import numpy as np
import pandas as pd

## For sampling trajectories and trees
from methods.seir import LogNormalSEIR, sample_traj
from methods.forest_sampling import random_rewiring_sampler

## For working with graphs
import networkx as nx

## For reproducibility, this ensures it's the
## same bundle of trajectories used throughout the 
## Washington COVID examples
np.random.seed(6)

if __name__ == "__main__":

	## Get the model outputs from prevalence.py
	fit_df = pd.read_pickle("..\\pickle_jar\\state_model.pkl")

	## Get the population information
	pyramid = pd.read_csv("..\\_data\\age_pyramid.csv")
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

	## Set up the sampling over a few trajectories or in
	## alignment with TransmissionForestVis.py. There are roughly
	## 10k trees per trajectory in the WA covid example, which
	## sets the number of trajectories needed for a target number of
	## trees.
	_version = 0
	if _version == 0:
		np.random.seed(2) ## for alignment with the second paper
		num_trajs = 25
		traj_ids = np.random.choice(new_exposures.index,
									replace=False,
									size=(num_trajs,))
	elif _version == 1:
		num_trajs = 1
		traj_ids = [908]
		np.random.seed(23) ## for alignment with the first paper

	## Loop over trajectories to sample an associated tree...
	comp_dfs = []
	print("\nStarting sampling...")
	for i,tid in enumerate(traj_ids):
		print("\nSample {}, trajectory {}".format(i,tid))

		## Pick a particular sample
		N_t = np.round(new_exposures.loc[tid]).astype(np.int32)

		## Compute the associated infections
		I_t = np.convolve(N_t.values,
						  np.array(model.D_i*[1]+model.D_e*[0]+[0]),
						  mode="same")
		I_t = pd.Series(I_t,index=N_t.index)
		I_t = I_t.shift(model.D_e+model.D_i).fillna(0).astype(np.int32)
		
		## To avoid edge effects, and the importations,
		## slice to mid February at least.
		N_t = N_t.loc["2020-02-25":"2021-03-15"]
		I_t = I_t.loc["2020-02-25":"2021-03-15"]

		## Get the daily degree distributions and align it
		## to the N_t series.
		pk_df = pd.read_pickle("..\\pickle_jar\\pk_timeseries.pkl")
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
	
		## Loop over days and construct constellations
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
		print("This forest sample:")

		## Check some statistics...
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

		## Compute the size distribution of the components and other
		## features of the components.
		comp_df = []
		components = nx.connected_components(undirected_G)
		for i,c in enumerate(components):

			## Get the node list together
			nodes_in_c = sorted(list(c))
			cdf = df.loc[nodes_in_c].copy()

			## Compute some transmission chain stats
			start_date = cdf["time"].min()
			end_date = cdf["time"].max()
			lifetime = (end_date-start_date).days
			size = len(cdf)
			
			## Store the results
			comp_df.append((size,start_date,end_date,lifetime))
			
		## Put it all together
		comp_df = pd.DataFrame(comp_df,
							   columns=["chain_size","start_date","end_date","chain_life"])
		print("Total number of components = {}".format(len(comp_df)))
		
		## Store the result
		comp_dfs.append(comp_df)
	
	## Put it together
	comp_df = pd.concat(comp_dfs,axis=0).reset_index(drop=True)
	print("\nTree ensemble dataset:")
	print(comp_df)
	comp_df.to_pickle("..\\pickle_jar\\sampled_trees.pkl")