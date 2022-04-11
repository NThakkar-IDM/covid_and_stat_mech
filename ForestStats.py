""" ForestStats.py

Use trajectories and distributions from IndividualLevelDistributions.py to sample
a plausible transmission tree, and then compare that tree's chain durations to figure 
4 here:
https://www.thelancet.com/journals/lanam/article/PIIS2667-193X(21)00010-7/fulltext  """
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

def GetPhyloData():

	## Uses pandas i/o
	df = pd.read_csv("_data\\datafor.WA.SARS.Tordoff.Figure4b.csv",
					 header=0,
					 usecols=["lineage","i","j","k",
					 		  "min.sts","max.sts",
					 		  "size"],
					 )
	df = df.iloc[1:].reset_index(drop=True) ## drop the blank row

	## Get the start and end dates
	df["t0"] = (pd.to_datetime("2020-01-01")+\
				pd.to_timedelta(365*(df["min.sts"]-2020),unit="d"))
	df["t1"] = (pd.to_datetime("2020-01-01")+\
				pd.to_timedelta(365*(df["max.sts"]-2020),unit="d"))
	df["duration"] = (df["t1"] - df["t0"]).dt.round("d").dt.days

	## Drop the redundant stuff and clean
	df = df.drop(columns=["min.sts","max.sts"])

	return df

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

	## From the model output (Prevalence.py), for reporting rate
	## adjustment.
	rep_rate = pd.read_pickle("pickle_jar\\model_rep_rate.pkl")
	rep_rate = rep_rate["avg_rr"]/100.

	## Set up the sampling over a few trajectories or in
	## alignment with TransmissionForestVis.py
	_version = 1
	if _version == 0:
		np.random.seed()#23)
		num_trees = 2
		traj_ids = np.random.choice(new_exposures.index,
									replace=False,
									size=(num_trees,))
	elif _version == 1:
		num_trees = 1
		traj_ids = [908]
		np.random.seed(23)

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
	
		## Loop over days
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

		## Compute the size distribution of the components, sampling
		## nodes in bernouilli trials.
		comp_df = []
		components = nx.connected_components(undirected_G)
		p_seq = 0.064 ## From Tordoff et al
		for i,c in enumerate(components):

			## Get the node list together
			if len(c) == 1:
				continue
			nodes_in_c = sorted(list(c))
			cdf = df.loc[nodes_in_c].copy()

			## Compute some transmission chain stats
			lifetime = (cdf["time"].max()-cdf["time"].min()).days
			size = len(cdf)
			
			## Subsample to a clade
			p_reported = rep_rate.loc[cdf["time"]]
			seq_nodes = np.random.binomial(1,p_seq*p_reported.values,
										   size=(len(nodes_in_c,)))
			clade = cdf.loc[seq_nodes.astype(bool)]

			## Compute some clade stats
			clade_size = len(clade)
			clade_life = (clade["time"].max()-clade["time"].min()).days

			## Store the results
			comp_df.append((size,lifetime,clade_size,clade_life))
			
		## Put it all together
		comp_df = pd.DataFrame(comp_df,
							   columns=["chain_size","chain_life","seqs","duration"])
		print("Total number of non-trivial components = {}".format(len(comp_df)))
		print("Total number of components at all = {}".format(i))

		## Store the result
		comp_dfs.append(comp_df)
	
	## Put it together
	comp_df = pd.concat(comp_dfs,axis=0).reset_index(drop=True)
	print("\nTree ensemble dataset:")
	print(comp_df)

	## Then bin the sizes similar to the paper
	bins = [-0.1,2,5,10,20,np.inf]
	labels = ["2","3 to 5","6 to 10","11 to 20","over 20"]
	comp_df = comp_df.loc[comp_df["seqs"] > 1]
	comp_df["size_bin"] = pd.cut(comp_df["seqs"],bins,labels=labels)

	## Get the phylo for comparsion
	phylo = GetPhyloData()
	phylo["size_bin"] = pd.cut(phylo["size"],bins,labels=labels)

	## Set up a plot
	fig, axes = plt.subplots(figsize=(10,6))
	axes_setup(axes)
	y_ticks = []
	y_labels = []
	for i, (b, sf) in enumerate(comp_df.groupby("size_bin")):
		if i == 0:
			label = "Model estimate"
		else:
			label = None
		axes.boxplot(sf["duration"].values,
					 vert=False,
					 positions=[2*i-0.25],
					 showfliers=False,
					 showcaps=False,
					 medianprops={"lw":2,"color":"#BF00BA","label":label},
					 boxprops={"color":"#BF00BA"},
					 whiskerprops={"color":"#BF00BA"},
					 flierprops={"markeredgecolor":"#BF00BA"},
					 )
		y_ticks.append(2*i)
		y_labels.append(b)
	for i, (b, sf) in enumerate(phylo.groupby("size_bin")):
		if i == 0:
			label="Phylogenetic estimate"
		else:
			label=None
		axes.boxplot(sf["duration"].values,
					 vert=False,
					 positions=[2*i+0.25],
					 showfliers=False,
					 showcaps=False,
					 medianprops={"lw":2,"color":"k","label":label},
					 boxprops={"color":"k"},
					 whiskerprops={"color":"k"},
					 flierprops={"markeredgecolor":"k"},
					 )
	axes.set_yticks(y_ticks)
	axes.set_yticklabels(y_labels)
	axes.set_xlim((0,None))
	axes.set_ylabel("Estimated clade size")
	axes.set_xlabel("Transmission chain lifetime (days)")
	axes.legend(loc=4,frameon=False,fontsize=18)
	fig.tight_layout()
	fig.savefig("_plots\\transmission_chain_duration_by_size.png")

	## What about the size distributions?
	phylo_hist = phylo["size_bin"].value_counts()/len(phylo)
	comp_hist = comp_df["size_bin"].value_counts()/len(comp_df)
	phylo_hist = phylo_hist.sort_index()
	comp_hist = comp_hist.sort_index()
	fig, axes = plt.subplots(figsize=(6,6))
	axes_setup(axes)
	x = np.arange(len(phylo_hist))
	w = 0.2
	axes.bar(x-0.5*w,phylo_hist.values,width=w,
			  color="k",label="Phylo")
	axes.bar(x+0.5*w,comp_hist.values,width=w,
			  color="#BF00BA",label="Model")
	axes.set_xticks(x)
	axes.set_xticklabels(comp_hist.index,rotation=45)
	axes.set_ylabel("Probability")
	axes.legend(loc=1,frameon=False)
	fig.tight_layout()
	fig.savefig("_plots\\phylo_and_tree_size_dists.png")

	## Finish
	plt.show()
