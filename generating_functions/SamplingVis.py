""" SamplingVis.py

Script to make figure 1 in the manucript, which visualizes a sample forest and
has some additional cartoons and checks."""
import sys
sys.path.append("..\\")
import methods

## standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For sampling trajectories and trees
from methods.seir import LogNormalSEIR, sample_traj
from methods.forest_sampling import random_rewiring_sampler

## For probability calculations
from scipy.special import gamma, gammaln, binom

## For working with graphs
import networkx as nx

## some colors
plt.rcParams["font.size"] = 22.
colors = {"blue":"#3079F7","red":"#F73079",
		  "yellow":"#F7AE30","purple":"#E030F7"}

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def gamma_poisson_density(x,m,k):
	ln_N = gammaln(x+k)-gammaln(x+1)-gammaln(k)
	ln_sf = x*np.log(m/(m+k))+k*np.log(k/(m+k))
	p_k = np.exp(ln_N+ln_sf)
	p_k *= 1./(p_k.sum())
	return p_k

if __name__ == "__main__":

	## Sample a reduced forest
	np.random.seed(6) 

	## Get the model outputs from prevalence.py
	fit_df = pd.read_pickle("..\\pickle_jar\\state_model.pkl")

	## Get the population information
	pyramid = pd.read_csv("..\\_data\\age_pyramid.csv")
	pyramid = pyramid.set_index("age_bin")["population"]
	population = pyramid.sum()

	## Create a model
	print("\nSampling a reduced forest...")
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

	## Pick a particular sample, and reduce the number of nodes
	## for visual clarity
	N_t = np.round(0.01*new_exposures.loc[1123]).astype(np.int32)

	## Compute the (approximate) associated infections
	I_t = np.convolve(N_t.values,
					  np.array(model.D_i*[1]+model.D_e*[0]+[0]), 
					  mode="same")
	I_t = pd.Series(I_t,index=N_t.index)
	I_t = I_t.shift(model.D_e+model.D_i).fillna(0).astype(np.int32)
	
	## To avoid edge effects, and the importations,
	## slice to mid February at least.
	N_t = N_t.loc["2020-02-25":"2021-01-01"]
	I_t = I_t.loc["2020-02-25":"2021-01-01"]

	## Get the daily degree distributions and align it
	## to the N_t series.
	pk_df = pd.read_pickle("..\\pickle_jar\\pk_timeseries.pkl")
	pk_df = pk_df.reindex(N_t.index).fillna(method="bfill")
	max_event_size = max(pk_df.columns[-1],N_t.max()+1)
	pk_df = pk_df.T.reindex(np.arange(0,max_event_size,dtype=np.int32)).fillna(0).T
	pk_df = pk_df.div(pk_df.sum(axis=1),axis=0)

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
	print("Forest sample:")
	print(df)
	
	## Make a huge figure
	np.random.seed(2)
	fig = plt.figure(figsize=(12,10))
	tree_ax = fig.add_subplot(2,2,(1,2))
	alg_ax = fig.add_subplot(2,2,(3,3))
	test_axes = [fig.add_subplot(2,2,(4,4))]

	## Make the cartoon
	alg_ax.axis("off")

	## Start with the initial condition
	for i in range(0,4):
		alg_ax.plot([-0.15,0.15],[-i,-i],
					lw=2,color="k",zorder=-1)
		alg_ax.plot([-0.15],[-i],
					ls="None",
					marker="o",markersize=12,
					markerfacecolor=colors["red"],markeredgecolor=colors["red"],
					markeredgewidth=2)
		alg_ax.plot([0.15],[-i],
					ls="None",
					marker="o",markersize=12,
					markerfacecolor="white",markeredgecolor=colors["blue"],
					markeredgewidth=2)	
	alg_ax.plot(4*[-0.15],-np.arange(i,i+4),
				ls="None",
				marker="o",markersize=12,
				markerfacecolor=colors["red"],markeredgecolor=colors["red"],
				markeredgewidth=2)

	## Then the rewiring
	for i in range(0,4):
		if i <= 2:
			alg_ax.plot([0.85,1.15],[0,-i],
						lw=2,color="k",zorder=-1)
		else:
			alg_ax.plot([0.85,1.15],[-i,-i],
						lw=2,color="k",zorder=-1)
		alg_ax.plot([0.85],[-i],
					ls="None",
					marker="o",markersize=12,
					markerfacecolor=colors["red"],markeredgecolor=colors["red"],
					markeredgewidth=2)
		alg_ax.plot([1.15],[-i],
					ls="None",
					marker="o",markersize=12,
					markerfacecolor="white",markeredgecolor=colors["blue"],
					markeredgewidth=2)	
	alg_ax.plot(4*[0.85],-np.arange(i,i+4),
				ls="None",
				marker="o",markersize=12,
				markerfacecolor=colors["red"],markeredgecolor=colors["red"],
				markeredgewidth=2)
	
	## Finally an arrow
	alg_ax.plot([0.3,0.7],[-3,-3],color="k",lw=2)
	alg_ax.plot([0.7],[-3],color="k",
				ls="None",marker=">",markersize=13)

	## Annotate
	alg_ax.text(0.5,-3.3,r"$\ell=3$",
				fontsize=18,color="k",
				horizontalalignment="center",verticalalignment="top")
	alg_ax.text(0.,-7.1,r"$(1,1,1,1,0,0,0)$",
				fontsize=18,color="k",
				horizontalalignment="center",verticalalignment="top")
	alg_ax.text(1.,-7.1,r"$(3,0,0,1,0,0,0)$",
				fontsize=18,color="k",
				horizontalalignment="center",verticalalignment="top")

	## Finish up
	alg_ax.set(xlim=(-0.4,1.7),
			   ylim=(-8.5,1.5))

	## Do some constellation sampling experiments for the
	## next panel
	Ns = np.array([27,54,12,38])
	Is = np.array([334,254,400,554])
	mus = Ns/Is
	ks = np.array([0.01,1.503687e-02,0.01,0.05,0.25,1.25])
	ls = np.arange(100)
	for i, ax in enumerate(test_axes):

		## Gather the distribution
		ls_given_N = ls[:Ns[i]+1]
		p_k = gamma_poisson_density(ls,mus[i],ks[i])
		p_k_given_N = p_k[:Ns[i]+1]/(1. - np.sum(p_k[Ns[i]+1:]))
		non_zero_p_k = p_k_given_N[1:]/(1. - p_k_given_N[0])

		## Draw samples
		samples = random_rewiring_sampler(Ns[i],
										  Is[i],
										  ls_given_N,
										  non_zero_p_k,
										  )
		samples = pd.DataFrame(samples)
		
		## Compute an overall histogram
		hist = samples.stack().reset_index(drop=True).value_counts().sort_index()
		hist = hist.reindex(ls_given_N).fillna(0)

		## Then compute the PMF estimate in a direchlet multinomial model
		alpha = hist + 1
		beta = alpha.sum() - alpha
		pmf = alpha/(alpha+beta)
		std = np.sqrt((alpha*beta)/(((alpha+beta)**2)*(alpha+beta+1)))

		## Plot it
		axes_setup(ax)
		#ax.grid(color="grey",alpha=0.2)
		#ax.plot(ls_given_N-0.5,p_k_given_N,lw=4,drawstyle="steps-post",
		#		color="k",label="Exact distribution")
		ax.bar(ls_given_N[:-1],p_k_given_N[:-1],
				width=1.,
				edgecolor="k",facecolor="None",lw=2,
				label="Exact distribution")
		ax.fill_between(pmf.index-0.5,
						pmf.values-2.*std.values,
						pmf.values+2.*std.values,
						edgecolor="None",facecolor=colors["purple"],alpha=0.2,
						step="post")
		ax.plot(pmf.index-0.5,pmf.values,lw=4,drawstyle="steps-post",
				color=colors["purple"],label="Sampled distribution")
		ax.set_xlabel("Transmission event size")
		ax.set_ylabel("Probability")
		ax.set_yscale("log")
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[::-1], labels[::-1],loc=1,frameon=False)

	## Return to the forest sample, and compute the 
	## profile
	df = df.loc[df["time"] <= "2020-10-01"]
	N_t = df["time"].value_counts().sort_index()

	## Make a graph object from it
	G = nx.DiGraph()
	for t, sf in df.groupby("time"):
		G.add_nodes_from(sf["node"],
						 time=t)
	edges = df.loc[df["parent"].notnull(),["parent","node"]]
	G.add_edges_from(edges.values)
	undirected_G = G.to_undirected()

	## Make another graph highlighting a particular day's
	## event constellation
	plot_date = N_t.index[0] + pd.to_timedelta(54,unit="d")
	day_t = df.loc[df["time"] == plot_date].copy()
	children = day_t["node"].copy()
	parents = df.loc[(df["inf_start"] <= plot_date) &\
					 (df["inf_end"] > plot_date),"node"].copy()
	C = nx.Graph()
	C.add_nodes_from(children,
					 nodetype="child")
	C.add_nodes_from(parents,
					nodetype="parent")
	C.add_edges_from(day_t[["parent","node"]].values)
	
	## Compute x-y positions (are there smarter ways to do this?)
	## First as a bar chart, then as a clock.
	bar_pos = {}
	for t, sf in df.groupby("time"):
		d = (t - N_t.index[0]).days
		this_day = {n:(d,i) for i,n in enumerate(sf["node"])}
		bar_pos.update(this_day)

	## Plot the bar graph version
	axes_setup(tree_ax)
	nx.draw_networkx_edges(undirected_G,
						   bar_pos, 
						   ax=tree_ax,
						   alpha=0.2,
						   edge_color="grey",
						   )
	nx.draw_networkx_edges(C,
						   bar_pos, 
						   ax=tree_ax,
						   alpha=0.9,
						   edge_color="k",
						   width=1,
						   )
	child_nodes = np.array([bar_pos[n] for n in children.values])
	tree_ax.plot(child_nodes[:,0],child_nodes[:,1],
				 marker="o",markeredgecolor=colors["blue"],markerfacecolor="white",
				 markersize=5,ls="None")
	parent_nodes = np.array([bar_pos[n] for n in parents.values])
	tree_ax.plot(parent_nodes[:,0],parent_nodes[:,1],
				 marker="o",markeredgecolor=colors["red"],markerfacecolor=colors["red"],
				 markersize=5,ls="None")
	
	## Overlay the time series
	tree_ax.plot(N_t.values,color="k",lw=3,zorder=0)
	
	## Set up the ticks
	tree_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
	tree_ax.set_ylabel(r"Daily infections, $N_t$")
	tree_ax.set_ylim((-0.75,None))
	tree_ax.set_xlabel("Time (days)")
	ticks = pd.date_range(N_t.index[0],N_t.index[-1],freq="MS")
	tree_ax.set_xlim((0,(pd.to_datetime("2020-07-01")-N_t.index[0]).days))

	## Adjust figure spacing
	fig.tight_layout()

	## Add the zoom inset
	x1 = (day_t["time"].min() - N_t.index[0]).days - 9
	x2 = (day_t["time"].max() - N_t.index[0]).days + 1
	y1 = tree_ax.get_ylim()[0]
	y2 = N_t.loc[day_t["time"].min()-pd.to_timedelta(9,unit="d"):
				 day_t["time"].max()+pd.to_timedelta(1,unit="d")].max()#+1
	const_ax = tree_ax.inset_axes([0.49, 0.31, 0.26, 0.76], #[0.49, 0.22, 0.26, 0.85],
			xlim=(x1, x2), ylim=(y1, y2), xticks=[], yticks=[])
	nx.draw_networkx_edges(undirected_G,
						   bar_pos, 
						   ax=const_ax,
						   alpha=0.2,
						   edge_color="grey",
						   )
	nx.draw_networkx_edges(C,
						   bar_pos, 
						   ax=const_ax,
						   alpha=0.9,
						   edge_color="k",
						   width=2,
						   )
	child_nodes = np.array([bar_pos[n] for n in children.values])
	const_ax.plot(child_nodes[:,0],child_nodes[:,1],
				 marker="o",markeredgecolor=colors["blue"],markerfacecolor="white",
				 markersize=8,ls="None",
				 markeredgewidth=2)
	parent_nodes = np.array([bar_pos[n] for n in parents.values])
	const_ax.plot(parent_nodes[:,0],parent_nodes[:,1],
				 marker="o",markeredgecolor=colors["red"],markerfacecolor=colors["red"],
				 markersize=8,ls="None")
	const_ax.plot(N_t.values,color="k",lw=4,zorder=0)
	tree_ax.indicate_inset_zoom(const_ax,edgecolor="k",lw=2)

	## Add panel labels
	alg_ax.text(-0.025,0.965,"b.",fontsize=20,color="k",transform=alg_ax.transAxes,verticalalignment="bottom")
	test_axes[0].text(-0.27,0.95,"c.",fontsize=20,color="k",transform=test_axes[0].transAxes,verticalalignment="bottom")
	tree_ax.text(-0.1,1.,"a.",fontsize=20,color="k",transform=tree_ax.transAxes)

	## Done
	plt.savefig("..\\_plots\\sampling_demo.png")
	plt.show()