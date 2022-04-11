""" PathogensisGRPVis.py

Signal processing of the case and hosp data to generate a heteroskedastic epicurve with timescale associated with
the pathogensis distribution. The variance estimates here are via GPR.

This script follows the first part of PrevalenceModel.py to generate the epi-curve, but instead of using
method from methods/pathogenesis.py, which is geared towards signal processing as a means to an end, this script
walks through the PathogenesisGRP function and visualizes key pieces (Figs 1 and 2 in the manuscript, for example). """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For time-series modeling and Reff estimation
from methods.severity import GaussianProcessIHR,\
							 ihr_table
from methods.vaccine import BetaVaccineImmunity,\
							BetaVaccineFailures
from methods.pathogenesis import AdjustedRWEpiCurve

## Some colors
colors = [
"#bd33ff",
"#ff33db",
"#ff3375",
"#ff5733",
"#ffbd33",
"#dbff33",
"#FFBD33",
]

## Hyper parameters and additional options
np.random.seed(6) ## To align the trajectory bundle
hp = {"tr_start":"2020-02-29",
	  "tr_end":"2021-03-26",
	  "unabated_date":"2020-03-01",
	  "gp_ifr":True,
	  "treatment_effect":True,
	  "symptom_suppression":False,
	  "plot_time_end":"2021-03-26",
	  "model_pickle":"state_model.pkl",
	  }

## Helper functions
def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

if __name__ == "__main__":
	
	## Compile the full time series data set, including the age
	## distribution of cases to determine the outcome probabilities
	## over time. 	
	raw_timeseries = pd.read_csv("_data\\epi_timeseries.csv")
	raw_timeseries["time"] = pd.to_datetime(raw_timeseries["time"])
	raw_timeseries = raw_timeseries.set_index("time")
	age_df = pd.read_csv("_data\\cases_by_age.csv")
	age_df["time"] = pd.to_datetime(age_df["time"])
	age_df = age_df.set_index("time")

	## Get the population information
	pyramid = pd.read_csv("_data\\age_pyramid.csv")
	pyramid = pyramid.set_index("age_bin")["population"]
	population = pyramid.sum()
	pyramid = pyramid/population

	## How do you handle data at the end, where increased testing and
	## lags might be an issue?
	timeseries = raw_timeseries.loc[:hp["tr_end"]].copy()
	time = pd.date_range(start="01-15-2020",end=timeseries.index[-1],freq="d")

	## Set up time range for plotting as well
	plot_time = pd.date_range(start=time[0],end=hp["plot_time_end"],freq="d")
	ticks = pd.date_range(plot_time[0],plot_time[-1],freq="MS")
	tick_labels = [t.strftime("%b")+(t.month == 1)*"\n{}".format(t.year) for t in ticks]

	## Prepare the raw vaccine derived immunity time series, for use
	## in Reff estimatation and model sampling.
	dose_df = pd.read_csv("_data\\vaccine_timeseries.csv")
	dose_df["time"] = pd.to_datetime(dose_df["time"])
	dose_df = dose_df.set_index("time")
	raw_vax = BetaVaccineImmunity(dose1=dose_df["dose1"],dose2=dose_df["dose2"])
	raw_vax = raw_vax.reindex(plot_time).fillna(method="bfill").fillna(method="ffill")
	
	## Set up treatment effects
	if hp["treatment_effect"]:
		treatment_factors = pd.read_csv("_data\\monthly_treatment_hazard.csv")
		treatment_factors["time"] = pd.to_datetime(treatment_factors["time"])
		treatment_factors = treatment_factors.set_index("time")
		treatment_factors.columns = ["ifr","ifr_var"]
		treatment_factors.loc[age_df.index[-1]] = [np.nan,np.nan]
		treatment_factors = treatment_factors.resample("d").asfreq().interpolate().reindex(age_df.index)
		treatment_factors["ihr"] = np.ones((len(treatment_factors),))
		treatment_factors["ihr_var"] = np.zeros((len(treatment_factors),))
	else:
		treatment_factors = pd.DataFrame(len(age_df.index)*[[np.nan,np.nan,np.nan,np.nan]],
										 index=age_df.index,columns=["ifr","ifr_var","ihr","ihr_var"])
		treatment_factors.loc[age_df.index[0]] = [1,0,1,0]
		treatment_factors = treatment_factors.interpolate()

	## For vaccine related suppression to the IHR, compute the total population with
	## vaccine failure, then compute the approximate fraction of susceptibles experiencing
	## symptom suppression, modifying "treatment-effects" accordingly.
	if hp["symptom_suppression"]:
		raw_failures = BetaVaccineFailures(dose1=dose_df["dose1"],dose2=dose_df["dose2"])
		raw_failures = raw_failures.reindex(raw_vax.index).fillna(method="bfill").fillna(method="ffill")
		exp_immune = np.cumsum(raw_vax["avg"]).values
		ss_frac = raw_failures.multiply(0.23/(population-exp_immune),axis=0)
		ss_frac["var"] *= 0.23/(population-exp_immune)
		ss_frac = ss_frac.reindex(treatment_factors.index).fillna(method="ffill")
		treatment_factors["ihr"] *= (1.-ss_frac["avg"]).values
		treatment_factors["ihr_var"] += ss_frac["var"].values
	
	## Sample hospitalizations using the same pattern we do for mortality
	## First by calculating the time-varying IHR
	prior_ihr, case_based_ihr, post_ihr = GaussianProcessIHR(age_df,pyramid,ihr_table)
	post_ihr["mean"] *= treatment_factors["ihr"].values
	post_ihr["var"] = (treatment_factors["ihr_var"]+treatment_factors["ihr"]**2)*\
					  (post_ihr["var"]+post_ihr["mean"]**2)-(treatment_factors["ihr"]*post_ihr["mean"])**2
	if hp["gp_ifr"]:
		ihr = post_ihr.copy()
	else:
		ihr = prior_ihr.copy()
	ihr.loc[ihr.index[0]-pd.to_timedelta(7,unit="d")] = prior_ihr.loc[prior_ihr.index[0]].values
	ihr = ihr.reindex(plot_time).interpolate(limit_area="inside").fillna(method="bfill").fillna(method="ffill")

	## Use the dataset to compute a testing-adjusted epicurve
	epi_curve, fmu, _ = AdjustedRWEpiCurve(timeseries,ihr,prior_ihr,
										   debug=False,end_index=None,
										   weekend_smooth=False,hosp_only=False,
										   correlation_time=28)

	## Reindex cases to harmonize all the different timeseries, and set up
	## the full processing problem.
	cases = timeseries["cases"].reindex(time).fillna(0)
	hosp = timeseries["hosp"].reindex(time).fillna(0)
	epi_curve = epi_curve.reindex(time).fillna(0)
	fmu = fmu.reindex(time)
	D_e, D_i = 5, 4
	
	## Construct the other curves
	phi = epi_curve.copy()
	
	## Construct eta, prop to E_t
	D1 = D_e*np.diag((-1.+(1./D_i))*np.ones((len(phi)-1,)))\
		   + D_e*np.diag(np.ones((len(phi)-2,)),k=1)
	D1 = np.hstack([D1,np.zeros((len(phi)-1,1))])
	D1[-1,-1] = D_e
	eta = pd.Series(np.dot(D1,phi.values),
					index=phi.index[:-1])

	## And xi, prop to N_t
	D2 = np.diag((-1.+(1./D_e))*np.ones((len(eta)-1,)))\
		   + np.diag(np.ones((len(eta)-2,)),k=1)
	D2 = np.hstack([D2,np.zeros((len(eta)-1,1))])
	D2[-1,-1] = 1.
	xi = pd.Series(np.dot(D2,eta.values),
					index=eta.index[:-1])

	## Set up the P matrix, via the pathogensis distribution
	alpha = 1.97
	lam = 0.11
	y = np.arange(0,len(phi))
	inc_dist = alpha*lam*((y*lam)**(alpha-1.))*np.exp(-(y*lam)**(alpha))
	inc_dist *= D_i/(inc_dist.sum())
	Ps = np.tril(np.array([np.roll(inc_dist,i) for i in np.arange(len(inc_dist))]).T)
	P = Ps[:,:-2]

	## Construct the pathogensis operator, and compute it's
	## SVD to estimate the effective rank
	A = np.dot(P,np.dot(D2,D1))
	L = A - np.eye(len(A))
	u, s, vt = np.linalg.svd(L)
	p_k = s/np.sum(np.abs(s))
	H = np.sum(-p_k*np.log(p_k))
	erank = np.exp(H)

	## Construct the approximation by taking the effective
	## null space and projecting accordingly
	range_space = vt[:int(np.ceil(erank)),:] 
	null_space = vt[int(np.ceil(erank)):,:]
	phi_hat = pd.Series(np.dot(np.dot(null_space,phi.values),null_space),
						index=phi.index,name="phi_hat")
	r_t = phi-phi_hat

	## An aggregated 3 panel viualization of the
	## pathogenesis basis
	fig = plt.figure(figsize=(13.5,7.5))
	dist_ax = fig.add_subplot(2,3,(1,1))
	svd_ax = fig.add_subplot(2,3,(4,4))
	mode_ax = fig.add_subplot(2,3,(2,6))

	## Plot the distribution
	y_short = np.arange(0,31)
	dist_short = alpha*lam*((y_short*lam)**(alpha-1.))*np.exp(-(y_short*lam)**(alpha))
	dist_short = dist_short/np.sum(dist_short)
	dist_ax.spines["left"].set_visible(False)
	dist_ax.spines["top"].set_visible(False)
	dist_ax.spines["right"].set_visible(False)
	dist_ax.grid(color="grey",alpha=0.2)
	dist_ax.fill_between(y_short,0,dist_short,facecolor="#188AFF",edgecolor="None",alpha=0.6)
	dist_ax.plot(y_short,dist_short,color="#188AFF",lw=6)
	dist_ax.set_ylim((0,None))
	dist_ax.set_yticks([])
	dist_ax.set_xticks([0,5,10,15,20,25,30])
	dist_ax.set_xlabel("Days from infection to symptoms")
	dist_ax.set_ylabel("Probability")

	## Then the singular value distribution
	axes_setup(svd_ax)
	svd_ax.grid(color="grey",alpha=0.2)
	svd_ax.plot(p_k,
				color=colors[3],
				lw=6)
	svd_ax.axvline(erank,
				   color=colors[4],
				   ls="dashed",lw=3)
	svd_ax.text(erank-15,p_k[0]*0.9,"Effective\n"+r"rank of $L$",
			  color=colors[4],fontsize=18,
			  horizontalalignment="right",verticalalignment="bottom")
	svd_ax.set_yscale("log")
	svd_ax.set_yticks([1e-5,1e-4,1e-3,1e-2])
	svd_ax.set_xticks([0,100,200,300,400])
	svd_ax.set_ylim((1e-5,None))
	svd_ax.set_ylabel(r"$p_l = \sigma_l/\Sigma_m|\sigma_m|$")
	svd_ax.set_xlabel(r"Singular value index, $l$")

	## Finally the modes
	mode_ax.spines["left"].set_visible(False)
	mode_ax.spines["top"].set_visible(False)
	mode_ax.spines["right"].set_visible(False)
	cmap = plt.get_cmap("magma")
	modes = [1,10,20,30,40]#,50,60]
	range_colors = [cmap(i) for i in np.linspace(0.1,0.4,len(modes))]
	null_colors = [cmap(i) for i in np.linspace(0.6,0.9,len(modes))]
	for i,v in enumerate(modes):
		mode_ax.plot(phi.index,
					 range_space[v]+i+1,
					 lw=1,c="tab:grey")#FF188A")#range_colors[i])
		mode_ax.plot(phi.index,
					 null_space[v]-i-1,
					 lw=1,c="k")#188AFF")#null_colors[i])
	mode_ax.set_ylim((None,len(modes)+1))
	mode_ax.set_yticks([])
	mode_ax.set_xticks(ticks)
	mode_ax.set_xticklabels(tick_labels)
	mode_ax.text(0.5,0.99,r"Modes in the effective range of $L$",
				 horizontalalignment="center",verticalalignment="top",
				 fontsize=20,color="tab:grey",transform=mode_ax.transAxes)
	mode_ax.text(0.5,0.49,r"Modes in the effective null space of $L$",
				 horizontalalignment="center",verticalalignment="top",
				 fontsize=20,color="k",transform=mode_ax.transAxes)
	
	## Finish up
	fig.tight_layout()
	dist_ax.text(-0.17,1.,"a.",fontsize=18,color="k",transform=dist_ax.transAxes)
	svd_ax.text(-0.17,1.,"b.",fontsize=18,color="k",transform=svd_ax.transAxes)
	mode_ax.text(-0.025,1.,"c.",fontsize=18,color="k",transform=mode_ax.transAxes)
	fig.savefig("_plots\\pathogensis_sp_overview.png")

	## Now estimate the variance, treating weekends as missing data
	## and using the rolling modification of C_t to fit H_t as a measure
	## of relative observation volatility.
	## First construct the weekend residual.
	missing = phi.index.weekday.isin({5,6})
	holidays = set([pd.to_datetime(d) for d in ["2020-01-01",
												"2020-01-20",
												"2020-02-17",
												"2020-05-25",
												"2020-07-03",
												"2020-09-07",
												"2020-11-11",
												"2020-11-26",
												"2020-12-24",
												"2020-12-25",
												"2021-01-01",
												"2021-01-18",
												"2021-02-15",
												]])
	holidays = phi.index.isin(holidays)
	missing = missing | holidays
	t = phi.loc[~missing].index
	rw_t = r_t.loc[~missing].values

	## Then get the observation variance matrix ready
	F = np.diag((fmu.values[~missing]**2))
	Finv = np.diag(1./np.diag(F))
	w = rw_t.var()
	
	## Modify the pathogensis operator to be weekend specific
	Pw = P[~missing,:]

	## Construct the GPR kernel modeling correlation at the
	## infectious duration timescale.
	s = (phi.index-phi.index[0]).days.astype(float).values[:-2]
	s = np.dot(s[:,np.newaxis],np.ones((len(s),1)).T)
	K = (w/(len(rw_t)+1))*np.exp(-((s - s.T)**2)/(2.*D_i*D_i))

	## Construct the residual correlation matrix
	## and use it to solve for the conditional mean estimate of
	## eps_t.
	M = np.dot(Pw,np.dot(K,Pw.T))+w*F
	Minv = np.linalg.inv(M)
	m_t = np.dot(np.dot(K,np.dot(Pw.T,Minv)),rw_t)
	Veps = K - np.dot(K,np.dot(Pw.T,np.dot(Minv,Pw),K))

	## Use eps_t to construct an estimate of the variance
	## (is there a better way to do this?)
	Eepseps = np.diag((np.diag(Veps) + m_t**2)/1.) ## marginal variance estimate.
	eps_t = 1.*m_t
	h_t = np.diag(Eepseps)

	## And project the solution to the I_t space with the
	## full projection matrix to get I_t's covariance matrix
	## and associated marginals.
	phi_cov = np.dot(np.dot(P,Eepseps),P.T)
	phi_std = np.sqrt(np.diag(phi_cov))

	## From which we can construct the full suite of estimates
	eta_hat = pd.Series(np.dot(D1,phi_hat.values),
						index=phi.index[:-1])
	eta_cov = np.dot(D1,np.dot(phi_cov,D1.T))
	eta_std = np.sqrt(np.diag(eta_cov))
	xi_hat = pd.Series(np.dot(D2,eta_hat.values),
					   index=eta_hat.index[:-1])
	xi_cov = np.dot(D2,np.dot(eta_cov,D2.T))
	xi_std = np.sqrt(np.diag(xi_cov))
	
	## Make a data to GRP plot
	_version = 1
	if _version == 0:
		fig, axes = plt.subplots(2,1,sharex=True,figsize=(13,7))
		for ax in axes:
			axes_setup(ax)

		## Plot the data panel, starting with cases
		axes[0].spines["left"].set_color("k")
		axes[0].plot(cases,color="k",lw=1)
		axes[0].set_ylabel(r"Daily cases, $C_t$",color="k")
		axes[0].tick_params(axis="y",colors="k")
		axes[0].set_ylim((0,None))

		## Then hosps on a separate axis
		hosp_ax = axes[0].twinx()
		hosp_ax.spines["left"].set_position(("axes",-0.15))
		hosp_ax.spines["top"].set_visible(False)
		hosp_ax.spines["right"].set_visible(False)
		hosp_ax.spines["bottom"].set_visible(False)
		hosp_ax.spines["left"].set_color(colors[2])
		hosp_ax.yaxis.set_label_position("left")
		hosp_ax.yaxis.set_ticks_position("left")
		hosp_ax.plot(hosp,color=colors[2],lw=1)
		hosp_ax.set_ylabel(r"Hosp. admissions, $H_t$",color=colors[2])
		hosp_ax.tick_params(axis="y",colors=colors[2])
		hosp_ax.set_ylim((0,None))

		## Plot the GRP panel
		axes[1].plot(phi,color="k",lw=1,zorder=1,label=r"Raw epi-curve, $f(\mu^*_t)C_t$")
		axes[1].fill_between(phi_hat.index,
							 phi_hat-2.*phi_std,
							 phi_hat+2.*phi_std,
							 facecolor="#188AFF",edgecolor="None",alpha=0.4,zorder=2)
		axes[1].plot(phi_hat,color="#188AFF",lw=2,zorder=3,label="Projection on the pathogenesis basis")
		axes[1].set_ylim((0,None))
		axes[1].set_ylabel(r"$\varphi_t$")
		axes[1].legend(loc=2,frameon=False)

		## Add the RW axis
		rw_ax = axes[0].twinx()
		rw_ax.spines["right"].set_position(("axes",1.025))
		rw_ax.spines["top"].set_visible(False)
		rw_ax.spines["left"].set_visible(False)
		rw_ax.spines["bottom"].set_visible(False)
		rw_ax.spines["right"].set_color("grey")
		weekly_tests = (timeseries["cases"]+timeseries["negatives"]).resample("W-SUN").sum()
		weekly_tests = weekly_tests.reindex(fmu.index).fillna(method="bfill").dropna()
		rw_ax.plot((weekly_tests/(weekly_tests.max()))*((1./fmu.values).max())*1.05,c="k",lw=2,ls="dashed",
					   label="Relative testing volume")
		rw_ax.fill_between(fmu.index,0,1./fmu.values,
						   facecolor="grey",edgecolor="None",alpha=0.25)
		rw_ax.set_ylabel(r"$f(\mu^*_t)$",color="grey")
		rw_ax.tick_params(axis="y",colors="grey")
		rw_ax.set_ylim((0,None))
		rw_ax.set_xticks(ticks)
		rw_ax.set_xticklabels(tick_labels)

		## Set the layering
		hosp_ax.set_zorder(rw_ax.get_zorder()+1)
		hosp_ax.patch.set_visible(False)
		axes[0].set_zorder(rw_ax.get_zorder()+2)
		axes[0].patch.set_visible(False)

		## Details
		axes[0].set_xlim((fmu.index[0],fmu.index[-1]))
		axes[1].set_xlim((fmu.index[0],fmu.index[-1]))
		fig.tight_layout()
		fig.savefig("_plots\\pathogensis_gpr_overview.png")

	elif _version == 1:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(13,9.7*0.97))
		for ax in axes:
			axes_setup(ax)

		## Plot the data panel, starting with cases
		axes[0].spines["left"].set_color("k")
		axes[0].plot(cases,color="k",lw=1)
		axes[0].set_ylabel(r"Daily cases, $C_t$",color="k")
		axes[0].tick_params(axis="y",colors="k")
		axes[0].set_yticks([0,1000,2000,3000,4000])
		axes[0].set_ylim((0,None))

		## Then hosps on a separate axis
		hosp_ax = axes[0].twinx()
		hosp_ax.spines["right"].set_position(("axes",1.025))
		hosp_ax.spines["top"].set_visible(False)
		hosp_ax.spines["left"].set_visible(False)
		hosp_ax.spines["bottom"].set_visible(False)
		hosp_ax.spines["right"].set_color("xkcd:cherry")
		hosp_ax.plot(hosp,color="xkcd:cherry",lw=1)
		hosp_ax.set_ylabel(r"Hosp. admissions, $H_t$",color="xkcd:cherry")
		hosp_ax.tick_params(axis="y",colors="xkcd:cherry")
		hosp_ax.set_ylim((0,None))

		## Plot the GRP panel
		axes[2].plot(phi,color="k",lw=1,zorder=1,label=r"Raw epi-curve, $f(\mu^*_t)C_t$")
		axes[2].fill_between(phi_hat.index,
							 phi_hat-2.*phi_std,
							 phi_hat+2.*phi_std,
							 facecolor="#BF00BA",edgecolor="None",alpha=0.4,zorder=2)
		axes[2].plot(phi_hat,color="#BF00BA",lw=2,zorder=3,label="Projection on the pathogenesis basis")
		axes[2].set_ylim((0,None))
		axes[2].set_ylabel(r"$\varphi_t$")
		axes[2].legend(loc=2,frameon=False)

		## set up the RW axis
		axes[1].spines["left"].set_color("xkcd:saffron")
		axes[1].plot(1./fmu,color="xkcd:saffron",lw=4)
		axes[1].set_ylabel(r"$1/f(\mu^*_t)$",color="xkcd:saffron")
		axes[1].tick_params(axis="y",colors="xkcd:saffron")
		axes[1].set_ylim((0,None))
		
		## Set up the testing panel
		t_ax = axes[1].twinx()
		t_ax.spines["right"].set_position(("axes",1.025))
		t_ax.spines["top"].set_visible(False)
		t_ax.spines["left"].set_visible(False)
		t_ax.spines["bottom"].set_visible(False)
		t_ax.spines["right"].set_color("grey")
		daily_tests = (timeseries["cases"]+timeseries["negatives"])#.rolling(7).sum()
		daily_tests = daily_tests.reindex(fmu.index).fillna(method="bfill").dropna()
		t_ax.fill_between(daily_tests.index,0,daily_tests.values,
						   facecolor="grey",edgecolor="None",alpha=0.25)
		t_ax.set_ylim((0,None))
		t_ax.set_ylabel(r"Daily tests",color="grey")
		t_ax.tick_params(axis="y",colors="grey")
		t_ax.set_xticks(ticks)
		t_ax.set_xticklabels(tick_labels)
		t_ax.set_yticks([0,10000,20000,30000,40000])

		## Set the layering
		axes[1].set_zorder(t_ax.get_zorder()+1)
		axes[1].patch.set_visible(False)

		## Details
		axes[0].set_xlim((fmu.index[0],fmu.index[-1]))
		axes[1].set_xlim((fmu.index[0],fmu.index[-1]))
		axes[2].set_xlim((fmu.index[0],fmu.index[-1]))
		fig.tight_layout()
		fig.savefig("_plots\\pathogensis_gpr_overview_v2.png")

	## Make a signal and noise plot
	fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,6))
	for ax in axes:
		axes_setup(ax)
	axes[0].plot(phi,color="k",alpha=0.3)
	axes[0].fill_between(phi_hat.index,
						 phi_hat-2.*phi_std,
						 phi_hat+2.*phi_std,
						 facecolor=colors[0],edgecolor="None",alpha=0.6)
	axes[0].plot(phi_hat,color=colors[0],lw=2)
	axes[1].fill_between(r_t.index,-2.*phi_std,2.*phi_std,
						 facecolor=colors[0],edgecolor="None",alpha=0.6)
	axes[1].plot(r_t,color="k",alpha=0.3,lw=1)
	axes[1].set_xticks(ticks)
	axes[1].set_xticklabels(tick_labels)
	axes[1].set_ylabel(r"$\varphi_t - \hat{\varphi}_t$")
	axes[0].set_ylabel(r"$\varphi_t$")
	fig.tight_layout()
	fig.savefig("_plots\\pathogensis_gpr.png")

	## Epi curves
	fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,8))
	for ax in axes:
		axes_setup(ax)
	axes[0].plot(phi,color="k",lw=1,alpha=0.3)
	axes[0].fill_between(phi_hat.index,
						 phi_hat-2.*phi_std,
						 phi_hat+2.*phi_std,
						 facecolor=colors[0],edgecolor="None",alpha=0.5)
	axes[0].plot(phi_hat,color=colors[0],lw=3)
	axes[1].plot(eta,color="k",lw=1,alpha=0.3)
	axes[1].fill_between(eta_hat.index,
						 eta_hat-2.*eta_std,
						 eta_hat+2.*eta_std,
						 facecolor=colors[1],edgecolor="None",alpha=0.5)	
	axes[1].plot(eta_hat,color=colors[1],lw=3)
	axes[2].plot(xi,color="k",lw=1,alpha=0.3)
	axes[2].fill_between(xi_hat.index,
						 xi_hat-2.*xi_std,
						 xi_hat+2.*xi_std,
						 facecolor=colors[2],edgecolor="None",alpha=0.5)	
	axes[2].plot(xi_hat,color=colors[2],lw=3)
	axes[0].set_ylabel(r"$\varphi_t$")
	axes[1].set_ylabel(r"$\eta_t$")
	axes[2].set_ylabel(r"$\xi_t$")
	axes[0].set_ylim((0,180))
	axes[1].set_ylim((0,220))
	axes[2].set_ylim((0,70))
	fig.tight_layout()

	## Done
	plt.show()
	