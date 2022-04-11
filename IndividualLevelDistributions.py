""" IndividualLevelDistributions.py

Compute the contact distributions for every day, save and visualze the outputs. """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6) ## To align the trajectory bundle

## For sampling, etc.
from methods.seir import LogNormalSEIR, sample_traj

## For densities
from methods.distributions import NegativeBinomial, PoissonDistribution

## Some colors
#colors = ["#00f40d","#006df4","#f400e7","#FF8D18","#f48700"]
colors = ["#3F681C","#791E94","#E71D36","#375E97"]

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

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

	## Compute summary stats
	exp_N = new_exposures.mean(axis=0).rename("exp_N")
	var_N = new_exposures.var(axis=0).rename("var_N")
	exp_I = infectious_pop.mean(axis=0).rename("exp_I")
	var_I = infectious_pop.var(axis=0).rename("var_I")

	## Compute the individual level summary stats
	exp_T = exp_N/exp_I
	var_T = (var_N-(exp_T**2)*var_I)/exp_I

	## Align with the data structure below
	t_df = pd.concat([exp_T.rename("avg"),
					  var_T.rename("var"),
					  exp_N,var_N,exp_I,var_I],axis=1).loc["2020-02-29":fit_df.index[-2]]
	t_df.to_pickle("pickle_jar\\T_dist_df.pkl")

	## And compute the standard error, interpolating
	## across places where the variance is undefined.
	t_df["bound"] = t_df["avg"]*(1-t_df["avg"])
	t_df.loc[t_df["var"] < t_df["avg"],"var"] = np.nan
	neg_vars = list(t_df.loc[t_df["var"].isnull()].index)
	
	## Compute negative binomial parameters via moment matching
	t_df["k"] = (t_df["avg"]**2)/(t_df["var"]-t_df["avg"])

	## For comparison purposes compute Reff
	fit_df["Reff"] = fit_df["beta_t"]*model.S0*model.D_i
	
	## Loop over days and store the inferred distributions
	pk_df = []
	f_ds = []
	errors = []
	print("\nStarting maximum entropy inference...")
	for d,r in t_df.iterrows():
		standard_med = NegativeBinomial(r["avg"],r["var"],N=1000)
		if standard_med.success:
			pk_df.append(standard_med.p_k)
		else:
			pk_df.append(standard_med.p_k)#*np.nan)
			f_ds.append(d)
			errors.append((np.sum(standard_med.p_k),
						   r["avg"]-standard_med.exp_k,
						   r["var"]-standard_med.var_k))
	
	## Error report
	print("\nDays with null var = {}".format(len(neg_vars)))
	print("Overall, failed convergence on {} days".format(len(f_ds)))
	errors = pd.DataFrame(errors,index=f_ds,
						  columns=["e{}".format(i) for i in range(3)])
	errors["interpolated_var"] = errors.index.isin(neg_vars)
	errors = pd.concat([errors,t_df.loc[errors.index,["avg","var","bound"]]],axis=1)
	print(errors)
	
	## Put together a data frame
	pk_df = pd.DataFrame(pk_df,
						 index=t_df.index,
						 columns=standard_med.k)#.interpolate()
	print("\nSaving this result:")
	print(pk_df)
	pk_df.to_pickle("pickle_jar\\pk_timeseries.pkl")

	## Aggregated overview plot, inluding special dates
	fig = plt.figure(figsize=(10,11))
	ss_ax = fig.add_subplot(3,2,(1,2))
	dist_axes = [fig.add_subplot(3,2,(3,3)),
				 fig.add_subplot(3,2,(4,4)),
				 fig.add_subplot(3,2,(5,5)),
				 fig.add_subplot(3,2,(6,6))]

	
	## Plot the ss probability
	axes_setup(ss_ax)
	ss_ax.plot(t_df["k"],
			   color="#FFBB00",
			   lw=3,zorder=4,label="Model estimate")
	ss_ax.axhline(0.1,color="grey",lw=1,zorder=2,label="Endo et al.")
	ss_ax.axhspan(0.05,0.2,
				  facecolor="grey",edgecolor="None",alpha=0.3,zorder=1)
	ss_ax.set_yscale("log")
	ss_ax.legend(loc=3,frameon=False)
	ss_ax.set_xticks(ticks)
	ss_ax.set_xticklabels(tick_labels)
	ss_ax.set_ylabel(r"Overdispersion, $k$")
	ss_ax.set_ylim((None,2))

	## Special dates
	for ax in dist_axes:
		axes_setup(ax)
		ax.grid(color="grey",alpha=0.15)
	dates = pd.to_datetime(["2020-03-04","2020-03-31","2020-07-04","2020-12-25"])
	date_labels = [d.strftime("%B ")+str(d.day)+d.strftime(", %Y") for d in dates]
	for i,ax in enumerate(dist_axes):
		med = pk_df.loc[dates[i]]
		ax.bar(med.index,med.values,1,edgecolor="None",facecolor=colors[i],alpha=1)
		ax.bar(med.index,med.values,1,edgecolor=colors[i],facecolor="None",lw=3)
		ax.set_xlim((-1,151))
		ax.text(0.98,0.95,
				date_labels[i],color=colors[i],fontsize=22,
				horizontalalignment="right",verticalalignment="top",
				transform=ax.transAxes)
		ax.set_yscale("log")
		ax.set_ylim((1e-6,1.))
		if i >= 2:
			ax.set_xlabel("Event size")
		if i%2 == 0:
			ax.set_ylabel("Probability")

		## Set up the ticks
		ax.tick_params(axis="y",which="minor",
					   reset=True,length=4.,width=2.)

	fig.tight_layout()
	ss_ax.text(-0.13,1.,"a.",fontsize=18,color="k",transform=ss_ax.transAxes)
	dist_axes[0].text(-0.30,1.,"b.",fontsize=18,color="k",transform=dist_axes[0].transAxes)
	fig.savefig("_plots\\gamma_poisson_overview.png")
	plt.show()
	sys.exit()
