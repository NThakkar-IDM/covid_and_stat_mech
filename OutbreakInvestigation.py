""" OutbreakInvestigation.py

Use the dataframes computed in IndividualLevelDistributions.py, in combination with some
outputs from PrevalenceModel.py to estimate the probability of a weekly outbreak report with
the model. """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(6)

## For sampling, etc.
from methods.seir import LogNormalSEIR, sample_traj

## Helper functions
def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def GetDecemberOutbreakReport(root="_data\\outbreak_report\\"):

	## Figure 1
	fig1 = []
	for line in open(root+"scanned_outbreak_report_20211214_fig1.dat"):
		if line.startswith("#"):
			continue
		fig1.append([np.float64(x) for x in line.split("\t")[:2]])
	fig1 = pd.DataFrame(fig1,columns=["x","y"])
	time = pd.date_range(start="2020-02-23",end="2021-12-31",freq="W-SUN")
	fig1["time"] = time[:len(fig1)]
	fig1["outbreaks"] = np.round(fig1["y"]).astype(np.int64)
	fig1 = fig1[["time","outbreaks"]].set_index("time")["outbreaks"]

	## Figure 2
	fig2 = []
	for line in open(root+"scanned_outbreak_report_20211214_fig2.dat"):
		if line.startswith("#"):
			continue
		fig2.append([np.float64(x) for x in line.split("\t")[:2]])
	fig2 = pd.DataFrame(fig2,columns=["x","y"])
	time = pd.date_range(start="2021-01-01",end="2021-12-31",freq="W-SUN")
	fig2["time"] = time[:len(fig2)]
	fig2["outbreaks"] = np.round(fig2["y"]).astype(np.int64)
	fig2 = fig2[["time","outbreaks"]].set_index("time")["outbreaks"]

	## Figure 3
	fig3 = []
	for line in open(root+"scanned_outbreak_report_20211214_fig3.dat"):
		if line.startswith("#"):
			continue
		fig3.append([np.float64(x) for x in line.split("\t")[:2]])
	fig3 = pd.DataFrame(fig3,columns=["x","y"])
	time = pd.date_range(start="2020-02-02",end="2021-12-31",freq="W-SUN")
	fig3["time"] = time[:len(fig3)]
	fig3["outbreaks"] = np.round(fig3["y"]).astype(np.int64)
	fig3 = fig3[["time","outbreaks"]].set_index("time")["outbreaks"]
	
	## Figure 4
	fig4 = []
	for line in open(root+"scanned_outbreak_report_20211214_fig4.dat"):
		if line.startswith("#"):
			continue
		fig4.append([np.float64(x) for x in line.split("\t")[:2]])
	fig4 = pd.DataFrame(fig4,columns=["x","y"])
	time = pd.date_range(start="2021-01-03",end="2021-12-31",freq="W-SUN")
	fig4["time"] = time[:len(fig4)]
	fig4["outbreaks"] = np.round(fig4["y"]).astype(np.int64)
	fig4 = fig4[["time","outbreaks"]].set_index("time")["outbreaks"]

	## Figure 5
	fig5 = []
	for line in open(root+"scanned_outbreak_report_20211214_fig5.dat"):
		if line.startswith("#"):
			continue
		fig5.append([np.float64(x) for x in line.split("\t")[:2]])
	fig5 = pd.DataFrame(fig5,columns=["x","y"])
	time = pd.date_range(start="2020-02-16",end="2021-12-31",freq="W-SUN")
	fig5["time"] = time[:len(fig5)]
	fig5["outbreaks"] = np.round(fig5["y"]).astype(np.int64)
	fig5 = fig5[["time","outbreaks"]].set_index("time")["outbreaks"]

	## Set up the high and low
	low = fig1.copy().rename("low")
	mid = pd.concat([fig1,fig4,fig5],axis=1)
	mid = mid.fillna(0).sum(axis=1).astype(np.int64).rename("mid")
	high = pd.concat([fig1,fig2,fig3,fig4,fig5],axis=1)
	high = high.fillna(0).sum(axis=1).astype(np.int64).rename("high")
	return low, mid, high

if __name__ == "__main__":

	## Get the summary statistics
	t_df = pd.read_pickle("pickle_jar\\T_dist_df.pkl")

	## Get the contact distributions
	pk_df = pd.read_pickle("pickle_jar\\pk_timeseries.pkl")
	
	## Get the model outputs from prevalence.py
	fit_df = pd.read_pickle("pickle_jar\\state_model.pkl")
	pyramid = pd.read_csv("_data\\age_pyramid.csv")
	pyramid = pyramid.set_index("age_bin")["population"]
	population = pyramid.sum()
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

	## Get the reporting metrics, starting with
	## the model-based estimate of the probability an infection
	## tests positive
	rep_rate = pd.read_pickle("pickle_jar\\model_rep_rate.pkl").loc[pk_df.index]
	rep_rate["avg_rr"] *= 1./100.
	
	## Then get the smoothed estimates of the fraction of
	## cases that get interviewed by contact tracers
	ct_df = pd.read_csv("_data\\fraction_of_cases_traced.csv")
	ct_df["time"] = pd.to_datetime(ct_df["time"])
	ct_df = ct_df.set_index("time")
	ct_df = ct_df.loc[pk_df.index]

	## Combine into a probability that an infection
	## is investigated.
	r_t = rep_rate["avg_rr"]*ct_df["fhat"]

	## Set up the probability of observing at least 2 of a k person
	## event.
	k = pk_df.columns.values
	r_t_k_2 = 1. - ((1.-r_t.values[:,np.newaxis])**(k[np.newaxis,:]-1))*(1+(k[np.newaxis,:]-1)*r_t.values[:,np.newaxis])
	print("\nEvent detection probability over time: ")
	print(pd.DataFrame(r_t_k_2,
					   index=pk_df.index,
					   columns=pk_df.columns))

	## So then, we can compute the probability of an observed outbreak event
	p_ss = pk_df*r_t_k_2
	p_ss = p_ss[np.arange(2,pk_df.columns[-1]+1)].sum(axis=1)
	fig, axes = plt.subplots(figsize=(12,6))
	axes_setup(axes)
	axes.plot(p_ss,color="k",lw=3)
	fig.tight_layout()
	
	## Calculate the expected number of outbreaks over time
	exp_ss = p_ss*t_df["exp_I"]
	var_o = p_ss*(1.-p_ss)
	var_ss = t_df["exp_I"]*var_o + (p_ss**2)*t_df["var_I"]

	## Convolve with the incubation distribution to incorporate
	## the time to symptom onset
	## Parameters from: https://advances.sciencemag.org/content/6/33/eabc1202
	alpha = 1.97
	lam = 0.11
	y = np.arange(0,31)
	inc_dist = alpha*lam*((y*lam)**(alpha-1.))*np.exp(-(y*lam)**(alpha))
	inc_dist = inc_dist/np.sum(inc_dist)
	z = np.arange(0,len(exp_ss))
	full_inc_dist = alpha*lam*((z*lam)**(alpha-1.))*np.exp(-(z*lam)**(alpha))
	full_inc_dist *= 1./(full_inc_dist.sum())
	Ps = np.tril(np.array([np.roll(full_inc_dist,i) for i in np.arange(len(full_inc_dist))]).T)
	exp_os = pd.Series(np.dot(Ps,exp_ss.values),
					   index=exp_ss.index)
	var_os = pd.Series(np.dot(Ps,var_ss.values),
					   index=exp_ss.index)
		
	## Diagonstic plot
	fig, axes = plt.subplots(figsize=(12,5))
	std_os = np.sqrt(var_os)
	prev = (samples[:,1,:]+samples[:,2,:]).mean(axis=0)
	prev *= exp_os.max()/(prev.max())
	axes.fill_between(exp_os.index,
					  (exp_os-2.*std_os).values,
					  (exp_os+2.*std_os).values,
					  facecolor="grey",edgecolor="None",alpha=0.6,zorder=3)
	axes.plot(fit_df.index,prev,color="xkcd:saffron",ls="dashed",alpha=1,lw=3,zorder=4)
	axes.plot(exp_os,color="k",lw=4,zorder=4,label="Prediction from the population-level model")
	fig.tight_layout()

	## Adjust to something closer to their "outbreak start date"
	mean_shift = 14
	exp_os.index = exp_os.index - pd.to_timedelta(int(mean_shift),unit="d")
	var_os.index = var_os.index - pd.to_timedelta(int(mean_shift),unit="d")

	## Get the outbreak data
	outbreaks_low, outbreaks_mid, outbreaks_high = GetDecemberOutbreakReport()
	outbreaks_low = outbreaks_low.loc[:exp_os.index[-1]]
	outbreaks_mid = outbreaks_mid.loc[:exp_os.index[-1]]
	outbreaks_high = outbreaks_high.loc[:exp_os.index[-1]]
	outbreaks = pd.concat([outbreaks_low,outbreaks_mid,outbreaks_high],axis=1).dropna()

	## Scale the estimate to account for house-hold events otherwise compatible 
	## with the definition of an outbreak
	scale_factor = np.sum((outbreaks["mid"].values*exp_os.loc[outbreaks.index].values))
	scale_factor *= 1./np.sum(exp_os.loc[outbreaks.index].values**2)
	exp_os *= scale_factor
	var_os *= scale_factor**2
	print("\nOverall scale factor = {}".format(scale_factor))
	print("So, weekly house-hold outbreaks = {}".format(1.-(scale_factor/7.)))
	
	## Compute the expected SS events over time
	std_ss = np.sqrt(var_ss)
	std_os = np.sqrt(var_os)
	
	## Plot the result
	fig, axes = plt.subplots(figsize=(12,5))
	axes_setup(axes)
	axes.errorbar(outbreaks.index,
				  outbreaks["mid"].values,
				  yerr = np.array([(outbreaks["mid"]-outbreaks["low"]).values,
				  				   (outbreaks["high"]-outbreaks["mid"]).values]),
				  lw=1,ls="None",color="k",zorder=10,label="The range across healthcare settings")
	axes.plot(outbreaks["mid"],color="k",ls="None",
			  marker="o",markersize=8,zorder=10,label="Weekly outbreak reports")
	axes.fill_between(exp_os.index,
					  (exp_os-2.*std_os).values,
					  (exp_os+2.*std_os).values,
					  facecolor="#4FB0C6",edgecolor="None",alpha=0.6,zorder=3)
	axes.plot(exp_os,color="#4F86C6",lw=4,zorder=4,label="Prediction from the population-level model")
	axes.set_ylim((0,None))
	axes.set_ylabel("Outbreak reports")
	ticks = pd.date_range("2020-03-01",pk_df.index[-1],freq="MS")
	tick_labels = [t.strftime("%b")+(t.month == 1)*"\n{}".format(t.year) for t in ticks]
	axes.set_xticks(ticks)
	axes.set_xticklabels(tick_labels)
	axes.legend(frameon=False,loc=2)
	fig.tight_layout()
	fig.savefig("_plots\\ss_outbreak_reports.png")

	## Done
	plt.show()