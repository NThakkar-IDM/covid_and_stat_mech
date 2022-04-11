""" ContactDistributionsVis.py

Analysis of the inferred, population-level transmission rate and volatility's 
implications for transmission at the individual level, visualized in a couple different
ways (Fig 4 in the manuscript). """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6)

## For sampling, etc.
from methods.seir import LogNormalSEIR, sample_traj

## For densities
from methods.distributions import NegativeBinomial

## Set up the pallette
#colors = ["#00ff07","#DF0000","#0078ff","#BF00BA"]
#colors = ["#00ff64","#001cff","#ff009b","#ffe300"]
colors = ["#00f40d","#006df4","#f400e7","#FF8D18","#f48700"]

## Helper functions
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

	## And compute the standard error, interpolating
	## across places where the variance is undefined.
	t_df["raw_var"] = t_df["var"].copy()
	t_df.loc[t_df["var"] < t_df["avg"],"var"] = np.nan
	neg_vars = list(t_df.loc[t_df["var"].isnull()].index)
	print("\nPlaces where variance estimation fails:")
	print(t_df.loc[neg_vars])
	t_df.loc[neg_vars,"var"] = t_df.loc[neg_vars,"avg"]
	t_df["std"] = np.sqrt(t_df["var"])

	## Compute negative binomial parameters via moment matching
	t_df["k"] = (t_df["avg"]**2)/(t_df["var"]-t_df["avg"])
	print(t_df)

	## For reference
	print("\nThe over-dispersion parameter...")
	print(t_df.loc[:"2020-03-22","k"].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))

	## When is variance low?
	dispersion = np.abs(t_df["raw_var"]/t_df["avg"])
	fig, axes = plt.subplots(figsize=(8,6))
	axes_setup(axes)
	axes.grid(color="grey",alpha=0.2)
	axes.plot(fit_df.loc[t_df.index,"sig_eps"].values,dispersion.values,
			  ls="None",marker="o",color="k",markersize=8)
	axes.plot(fit_df.loc[neg_vars,"sig_eps"].values,dispersion.loc[neg_vars].values,
			  ls="None",marker="o",markersize=12,
			  markerfacecolor="None",markeredgecolor="C3")
	axes.set(xlabel=r"Population-level volatility, Var$[$ln$\varepsilon_t]$",ylabel=r"Var$[T]$/Exp$[T]$")
	axes.set_yscale("log")
	fig.tight_layout()
	fig.savefig("_plots\\volatility_pop_to_person.png")
	
	## NaN bad estimates in the raw variance
	t_df.loc[neg_vars,"raw_var"] = np.nan
	
	## Plot densities
	xd = np.arange(0,40+1)
	date = pd.to_datetime("2020-03-10") 
	i1 = (date-t_df.index[0]).days #0

	## Compute some distributions for comparison
	med = NegativeBinomial(t_df["avg"].values[i1],t_df["var"].values[i1])
	print("\nOn {}...".format(date.strftime("%b %d, %Y")))
	expI_d = exp_I.loc[date]
	stdI_d = np.sqrt(var_I.loc[date])
	print("I_t = {} ({},{})".format(expI_d,expI_d-2*stdI_d,expI_d+2*stdI_d))
	expN_d = exp_N.loc[date]
	stdN_d = np.sqrt(var_N.loc[date])
	print("N_t = {} ({},{})".format(expN_d,expN_d-2*stdN_d,expN_d+2*stdN_d))

	## Plot the results
	fig, axes = plt.subplots(figsize=(8,7))
	axes_setup(axes)
	axes.grid(color="grey",alpha=0.15)
	axes.bar(med.k,med.p_k,1,lw=2,#drawstyle="steps-mid",
				facecolor=colors[3],edgecolor=colors[3],alpha=1)
	axes.set_xlim((-1,101))
	axes.set_xlabel("Event size")
	axes.text(0.98,0.95,
			  date.strftime("%B %d, %Y"),color=colors[3],fontsize=28,
			  horizontalalignment="right",verticalalignment="top",
			  transform=axes.transAxes)
	axes.axvline(32.5,ymax=0.6,color="k",ls="dashed",lw=3)
	axes.text(32.5/102.,0.6,
			  r"  $p(T = 32) \approx\,$"+"{0:0.4f}".format(med.p_k[31]),
			  color="k",fontsize=24,
			  horizontalalignment="left",verticalalignment="bottom",
			  transform=axes.transAxes)
	axes.set_yscale("log")
	axes.set_ylim((1e-6,1.))
	fig.tight_layout()
	fig.savefig("_plots\\event_distribution.png")

	## Big figure, with summary stats and distributions
	fig = plt.figure(figsize=(12,6.25*1.5))
	ss_ax = fig.add_subplot(4,4,(1,8))
	r_ax = fig.add_subplot(4,4,(9,12))
	dist_axes = [fig.add_subplot(4,4,(13,13)),
				 fig.add_subplot(4,4,(14,14)),
				 fig.add_subplot(4,4,(15,15)),
				 fig.add_subplot(4,4,(16,16))]

	## Plot the underlying event distribution
	axes_setup(ss_ax)
	ss_ax.fill_between(t_df.index,
					  np.clip(t_df["avg"].values-2.*t_df["std"].values,0,None),
					  t_df["avg"].values+2.*t_df["std"].values,
					  facecolor="#FF188A",edgecolor="None",alpha=0.4)#,label=r"95% confidence interval
	ss_ax.fill_between(t_df.index,
					  np.clip(t_df["avg"].values-2.*t_df["std"].values,0,None),
					  t_df["avg"].values+1.*t_df["std"].values,
					  facecolor="#FF188A",edgecolor="None",alpha=0.6,label="Standard deviation in that distribution")
	ss_ax.plot(t_df["avg"],lw=4,color="#188AFF",label="Average daily exposures per infectious person")

	## Add the prevalence context line
	prevalence = pd.DataFrame(100*(samples[:,1,:]+samples[:,2,:])/model.S0,
							  columns=fit_df.index)
	prev = prevalence.values.mean(axis=0)
	prev *= (t_df["avg"].values+2.*t_df["std"].values).max()/(prev.max())
	ss_ax.plot(fit_df.index,prev,color="k",ls="dashed",alpha=0.75,lw=3,label="Estimated COVID prevalence for context")

	## Highlight neg vars?
	ss_ax.plot(t_df.loc[neg_vars].index,
				 np.clip((t_df["avg"]+3.*t_df["std"]).loc[neg_vars].values,0,None),
				 ls="None",marker="o",markersize=10,markeredgewidth=2,
				 markerfacecolor="None",markeredgecolor="k")

	## Details
	ss_ax.set_ylabel("Transmission event size")
	ss_ax.set(ylim=(0,None),
				xlim=(t_df.index[0],t_df.index[-1]))
	ticks = pd.date_range("2020-03-01",t_df.index[-1],freq="MS")
	tick_labels = [t.strftime("%b")+(t.month == 1)*"\n{}".format(t.year) for t in ticks]
	ss_ax.set_xticks(ticks)
	ss_ax.set_xticklabels(tick_labels)
	h,l = ss_ax.get_legend_handles_labels()
	h = [h[0], h[2], h[1]]
	l = [l[0], l[2], l[1]]
	ss_ax.legend(h,l,loc=2,frameon=False,fontsize=20)

	## Plot the dispersion
	r_ax.spines["left"].set_position(("axes",-0.025))
	r_ax.spines["top"].set_visible(False)
	r_ax.spines["right"].set_visible(False)
	r_ax.grid(color="grey",alpha=0.15)
	r_ax.plot(t_df["k"],
			  color=colors[3],
			  lw=4)
	r_ax.set_yscale("log")
	r_ax.set_xticks(ticks)
	r_ax.set_xticklabels(tick_labels)
	r_ax.set_ylabel(r"$k_t$")
	r_ax.set_xlim((t_df.index[0],t_df.index[-1]))
	r_ax.set_yticks([1e-4,1e-3,1e-2,1e-1])

	## plot the dists
	for ax in dist_axes:
		axes_setup(ax)
		ax.grid(color="grey",alpha=0.15)
	dates = pd.to_datetime(["2020-03-04","2020-03-31","2020-07-04","2020-12-25"])
	date_labels = [d.strftime("%B ")+str(d.day)+d.strftime(", %Y") for d in dates]
	for i,ax in enumerate(dist_axes):
		j = (dates[i]-t_df.index[0]).days
		med = NegativeBinomial(t_df["avg"].values[j],
									  t_df["var"].values[j])
		ax.bar(med.k,med.p_k,1,edgecolor="None",facecolor="#8D18FF",alpha=1)
		ax.bar(med.k,med.p_k,1,edgecolor="#8D18FF",facecolor="None",lw=2)
		ax.set_xlim((-1,151))
		ax.set_xticks([0,50,100,150])
		ax.set_xlabel("Infectious contacts")
		if i == 0:
			ax.set_ylabel("Probability")
		ax.text(0.98,0.95,
				date_labels[i]+"\n"+r"$k_t=\,$"+"{0:0.3f}".format(t_df["k"].values[j]),
				color="k",fontsize=16,
				horizontalalignment="right",verticalalignment="top",
				transform=ax.transAxes)
		ax.set_yscale("log")
		ax.set_ylim((1e-6,1.))
		ax.set_yticks([1e-6,1e-3,1])

	## Finish up
	fig.tight_layout()
	fig.savefig("_plots\\t_over_time.png")
	plt.show()