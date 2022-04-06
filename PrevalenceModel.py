""" PrevalenceModel.py

An attempt to write a (geographically) general prevalence estimation pipeline.
This mostly follows the order in the manuscript: The data is processed via pathogenesis, the epi-curve
is fit to deaths, and then used to make SEIR transmission rates beta_t and epsilon_t. Along the way, the
epi-curve is used to estimate the reporting model parameters, and the pieces needed to reproduce the trajectory
bundle elsewhere are serialized. 

Hyper parameters are set in json style at the top. """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For time-series modeling and Reff estimation
from methods.seir import LogNormalSEIR,\
						 sample_traj,\
						 SampleOutcome
from methods.severity import GaussianProcessIFR,\
							 GaussianProcessIHR,\
							 ifr_table, ihr_table
from methods.vaccine import BetaVaccineImmunity,\
							BetaVaccineFailures
from methods.pathogenesis import AdjustedRWEpiCurve,\
								 PathogenesisGPR 

## For log normals and normals
from scipy.special import erf

## For checking performance
from sklearn.metrics import r2_score

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

def low_mid_high(samples):
	l0 = np.percentile(samples,1.,axis=0)
	h0 = np.percentile(samples,99.,axis=0)
	l1 = np.percentile(samples,2.5,axis=0)
	h1 = np.percentile(samples,97.5,axis=0)
	l2 = np.percentile(samples,25.,axis=0)
	h2 = np.percentile(samples,75.,axis=0)
	return l0, h0, l1, h1, l2, h2

def fit_quality(samples,data_col,verbose=True):

	## Compute a summary
	mean_x = samples.mean(axis=0)
	_, _, l1, h1, l2, h2 = low_mid_high(samples)

	## Align the lengths
	obs_x = raw_timeseries.loc[plot_time[0]:,data_col].values[1:]
	end_point = min(len(obs_x),len(mean_x))
	obs_x = obs_x[:end_point]

	## Compute scores
	score = r2_score(obs_x[:end_point],mean_x[:end_point])
	score95 = len(obs_x[np.where((obs_x >= l1) & (obs_x <= h1))])/end_point
	score50 = len(obs_x[np.where((obs_x >= l2) & (obs_x <= h2))])/end_point	
	if verbose:
		print("R2 score = {}".format(score))
		print("Within 50 interval: {}".format(score50))
		print("With 95 interval: {}".format(score95))
	return score, score50, score95

def log_normal_cdf(x,mu,sig):
	return 0.5 + 0.5*erf((np.log(x)-mu)/(sig*np.sqrt(2)))

def normal_cdf(x,mu,sig):
	return 0.5 + 0.5*erf((x-mu)/(sig*np.sqrt(2)))

## Output comparisons
def GetSCANPrevalence(version="recent"):

	if version == "recent":
		df = pd.read_csv("_data\\scanprev_5_21.csv",
						 usecols=["mean","lower","upper","since","to"])
		df = df[["mean","lower","upper","since","to"]]
		df.columns = ["mean","low","high","t0","t1"]
		df[["mean","low","high"]] = 100*df[["mean","low","high"]]
		df = df[["t0","t1","mean","low","high"]]
		scan_result = [tuple(r.values) for i,r in df.iterrows()]

	elif version == "published":
		scan_result = [("2020-03-23","2020-03-29",0.32,0.08,1.18),
					   ("2020-03-29","2020-04-04",0.27,0.07,0.95),
					   ("2020-04-04","2020-04-10",0.07,0.01,0.36)]

	elif version == "lancet":
		scan_result = [("2020-02-24","2020-03-09",100*6748/population,100*4133/population,100*11020/population)]

	return scan_result

def GetRtLive():
	df = pd.read_csv("_data\\rt.csv",
					 header=0,
					 usecols=["date","region",
					 		  "mean","median","lower_80","upper_80"])
	df = df.loc[df["region"] == "WA"]
	df = df.drop(columns=["region"])
	df["date"] = pd.to_datetime(df["date"])
	df = df.set_index("date")
	return df

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

	## Compute the IFR for this population based on age-structed estimates
	## from Verity et al. Compute distributions via the method in rainier.rainier
	prior_ifr, case_based_ifr, post_ifr = GaussianProcessIFR(age_df,pyramid,ifr_table)
	post_ifr["mean"] *= treatment_factors["ifr"].values
	post_ifr["var"] = (treatment_factors["ifr_var"]+treatment_factors["ifr"]**2)*\
					  (post_ifr["var"]+post_ifr["mean"]**2)-(treatment_factors["ifr"]*post_ifr["mean"])**2
	if hp["gp_ifr"]:
		ifr = post_ifr.copy()
	else:
		ifr = prior_ifr.copy()
	ifr.loc[ifr.index[0]-pd.to_timedelta(7,unit="d")] = prior_ifr.loc[prior_ifr.index[0]].values
	ifr = ifr.reindex(time).interpolate(limit_area="inside").fillna(method="bfill").fillna(method="ffill")
	
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

	## Reindex cases to harmonize all the different timeseries.
	cases = timeseries["cases"].reindex(time).fillna(0)
	deaths = timeseries["deaths"].reindex(time).fillna(0)
	hosp = timeseries["hosp"].reindex(time).fillna(0)
	epi_curve = epi_curve.reindex(time).fillna(0)
	fmu = fmu.reindex(time)

	## Set up empty storage for importations
	importations = pd.Series(np.zeros(len(cases),),
							 index=cases.index,
							 name="importations")

	## Set up the transmission regression start
	tr_date = pd.to_datetime(hp["tr_start"])
	tr_start = (tr_date-time[0]).days

	## Set up a model class to store relevant parameters
	## organize model fitting. 
	model = LogNormalSEIR(S0=population,
						  D_e=5,
						  D_i=4,
						  z_t=importations.values)
	
	## Pathogenesis GPR
	print("\nEstimating pathogensis projection...")
	phi_hat, phi_cov, eta_hat, eta_cov, xi_hat, xi_cov = PathogenesisGPR(model,
																		 epi_curve,
																		 fmu,
																		 debug=False)

	## Set up the mortality propogator to estimate the proportionality contstant
	z = np.arange(1e-5,len(time)+1)
	cdf = log_normal_cdf(z,2.8329,0.42)
	dist = cdf[1:]-cdf[:-1]
	dist = dist/np.sum(dist)
	mP = np.tril(np.array([np.roll(dist,i) for i in np.arange(len(dist))]).T)

	## And the hosp propogator to estimate the IHR constant
	cdf = normal_cdf(z,model.D_e+2.1,2.65)
	dist = cdf[1:]-cdf[:-1]
	dist = dist/np.sum(dist)
	hP = np.tril(np.array([np.roll(dist,i) for i in np.arange(len(dist))]).T)

	## And solve the least squares problem
	delta = np.dot(mP[:,:-2],xi_hat)
	dcov = np.dot(mP[:,:-2],np.dot(xi_cov,mP[:,:-2].T))
	vdelta = np.diag(dcov)
	sdelta = np.sqrt(vdelta)
	delta_adj = delta*(ifr["mean"]/100.)
	vdelta_adj = vdelta*((ifr["mean"]/100.)**2)
	sdelta_adj = sdelta*(ifr["mean"]/100.)
	c = np.sum(delta_adj*deaths)/(np.sum(delta_adj**2))
	print("...proportionality constant = {}".format(c))
	fig, axes = plt.subplots(figsize=(12,5))
	axes_setup(axes)
	axes.fill_between(time,
					  c*delta_adj-2.*c*sdelta_adj,
					  c*delta_adj+2.*c*sdelta_adj,
					  facecolor="C1",edgecolor="None",alpha=0.4)
	axes.plot(time,c*delta_adj,color="C1",lw=3)
	axes.plot(deaths,color="k",alpha=0.8)
	fig.tight_layout()

	## And construct full estimates
	Ehat = c*eta_hat
	Ecov = c*c*eta_cov
	Ihat = c*phi_hat
	Icov = c*c*phi_cov
	Nhat = c*xi_hat
	Ncov = c*c*xi_cov

	## And compute the IHR constant in a similar fashion
	delta = np.dot(hP[:,:-2],Nhat)
	dcov = np.dot(hP[:,:-2],np.dot(Ncov,hP[:,:-2].T))
	vdelta = np.diag(dcov)
	sdelta = np.sqrt(vdelta)
	delta_adj = delta*(ihr["mean"]/100.)
	vdelta_adj = vdelta*((ihr["mean"]/100.)**2)
	sdelta_adj = sdelta*(ihr["mean"]/100.)
	ihr_scale_factor = np.sum(delta_adj*hosp)/(np.sum(delta_adj**2))
	ihr["mean"] *= ihr_scale_factor
	ihr["var"] *= ihr_scale_factor**2
	ihr["std"] = np.sqrt(ihr["var"])
	print("...IHR scale factor = {}".format(ihr_scale_factor))
	fig, axes = plt.subplots(figsize=(12,5))
	axes_setup(axes)
	axes.fill_between(time,
					  ihr_scale_factor*delta_adj-2.*ihr_scale_factor*sdelta_adj,
					  ihr_scale_factor*delta_adj+2.*ihr_scale_factor*sdelta_adj,
					  facecolor="C3",edgecolor="None",alpha=0.4)
	axes.plot(time,ihr_scale_factor*delta_adj,color="C3",lw=3)
	axes.plot(hosp,color="k",alpha=0.8)
	fig.tight_layout()

	## And then compute the reporting rate
	fig, axes = plt.subplots(figsize=(12,6))
	rr_pe = (cases/Ihat)#.loc["2020-02-29":]
	X = np.array([1./fmu.loc[rr_pe.index].values,
				 (rr_pe.index.weekday.isin({5,6})).astype(np.float64)]).T
	X[:,1] *= X[:,0]
	C = np.linalg.inv(np.dot(X.T,X))
	p = np.dot(C,np.dot(X.T,rr_pe))
	rr = np.dot(X,p)
	pcov = C*(np.sum((rr_pe - rr)**2)/len(rr))
	rr_var = np.diag(np.dot(X,np.dot(pcov,X.T)))
	rr_std = np.sqrt(rr_var)
	axes_setup(axes)
	axes.plot(rr_pe,color="k",lw=1,ls="dashed")
	axes.fill_between(rr_pe.index,
					  rr-2.*rr_std,
					  rr+2.*rr_std,
					  facecolor="C4",edgecolor="None",alpha=0.3)
	axes.plot(rr_pe.index,rr,color="C4",lw=3)
	fig.tight_layout()

	## Construct the susceptible estimate
	avg_vax = raw_vax["avg"].loc[epi_curve.index].values[1:]
	cov_vax = np.diag(raw_vax["var"].loc[epi_curve.index].values[1:])
	cum_sum = np.tril(np.ones((len(model.time)-2,len(model.time)-2)))
	Shat = model.S0 - np.dot(cum_sum,Nhat)
	Scov = np.dot(cum_sum,np.dot(Ncov,cum_sum.T))
	avg_frac_susceptible = Shat/model.S0
	avg_vax = avg_frac_susceptible*avg_vax[:len(Nhat)]
	cov_vax = np.diag(avg_frac_susceptible**2)*(cov_vax[:len(Nhat),:len(Nhat)]) ## NB: this forces cov_vax to be diagonal
	Shat = pd.Series(Shat - np.dot(cum_sum,avg_vax),index=Nhat.index)
	Scov = Scov + np.dot(cum_sum,np.dot(cov_vax,cum_sum.T))
	
	## And compute log transmission rates and volatility
	lnbeta = np.log(Nhat.values[tr_start:]) \
		   - np.log(Shat.values[tr_start:]) \
		   - np.log(Ihat.values[tr_start:-2])
	lnbeta_var = np.diag(Ncov)[tr_start:]/(Nhat.values[tr_start:]**2)\
				 + np.diag(Scov)[tr_start:]/(Shat.values[tr_start:]**2)\
				 + np.diag(Icov)[tr_start:-2]/(Ihat.values[tr_start:-2]**2)

	## For conversion to an SEIR model 
	beta_t = pd.Series(lnbeta-0.5*lnbeta_var,index=time[tr_start:tr_start+len(lnbeta)])
	sig_eps = pd.Series(np.sqrt(lnbeta_var),index=time[tr_start:tr_start+len(lnbeta)])
	model.sig_eps = sig_eps.values

	## Compute R0 point estimates
	R0_point_est = np.exp(beta_t.values)*model.S0*model.D_i
	R0_point_est_std = np.exp(beta_t.values)*sig_eps.values*model.S0*model.D_i
	r0_estimates = pd.DataFrame(np.array([R0_point_est,R0_point_est_std]).T,
								columns=["r0_t","std_err"],
								index=time[tr_start:tr_start+len(R0_point_est)])
	print("\nPoint estimates for R0:")
	print(r0_estimates)

	## Backfill to set the initial conditions
	early_beta_t = beta_t.loc[:hp["unabated_date"]].mean()
	beta_t = beta_t.reindex(time).fillna(method="ffill").fillna(early_beta_t)
	sig_eps = sig_eps.reindex(time).fillna(method="ffill").fillna(sig_eps.mean())

	## Set up the IC eigenvalue problem
	Rt0 = np.exp(early_beta_t)*model.S0 
	A = np.array([[1.-(1./model.D_e),Rt0],
				  [1./model.D_e,1.-(1./model.D_i)]])
	w, v = np.linalg.eig(A)
	vinv = np.linalg.inv(v)
	dt = (r0_estimates.index[0]-time[0]).days
	target = np.array([Ehat.iloc[dt],Ihat.iloc[dt]])
	target_std = np.sqrt(np.array([Ecov[dt,dt],Icov[dt,dt]]))
	x0 = np.array([0.,1.])
	W = np.diag(w**dt)
	x1 = np.dot(v,np.dot(W,np.dot(vinv,x0)))
	pulse = x0*(target/np.dot(v,np.dot(W,np.dot(vinv,x0))))
	pulse_std = x0*(target_std/np.dot(v,np.dot(W,np.dot(vinv,x0))))
	print("\nInitial pulse:")
	print(pulse)
	print(pulse_std)
	
	## Sample some trajectories
	num_samples = 10000
	t0 = (beta_t.index[0] - Ihat.index[0]).days
	ic = np.array([np.random.normal(Shat.iloc[t0],np.sqrt(Scov[t0,t0]),size=(num_samples,)),
				   np.random.normal(pulse[0],pulse_std[0],size=(num_samples,)),
				   np.random.normal(pulse[1],pulse_std[1],size=(num_samples,))]).T
	ic = np.clip(ic,0,None)
	samples = sample_traj(model,
						  beta_t.values,
						  sig_eps=sig_eps.values,
						  z_t=importations.loc[beta_t.index].values,
						  avg_v_t=raw_vax.loc[beta_t.index,"avg"].values,
						  cov_v_t=np.diag(raw_vax.loc[beta_t.index,"var"].values),
						  ic=ic,
						  )

	## Sample cases
	rr = pd.Series(rr,index=rr_pe.index)
	case_samples = np.random.binomial(np.round(samples[:,2,:]).astype(int),
									  p=np.clip(rr.values,0,None))

	## Compute active infections (and describe)
	prevalence = pd.DataFrame((samples[:,1,:] + samples[:,2,:]).T/population,
							  index=plot_time).T
	print("\nPrevalence:")
	print(prevalence[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))
	immunity = pd.DataFrame((model.S0 - samples[:,0,:]).T/population,
							index=plot_time).T
	print("\nOverall immunity:")
	print(immunity[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))

	## For downstream estimates, compute daily new exposures from the slope of the
	## exposed compartment.
	new_exposures = samples[:,1,1:] - (1. - (1./model.D_e))*samples[:,1,:-1]
	new_exposures = pd.DataFrame(new_exposures.T,
								 index=time[:-1]).reindex(time).fillna(0).T

	## Compute the cumulative reporting rate
	total_cases = timeseries["cases"].sum()
	cum_rr_samples = 100*total_cases/np.cumsum(new_exposures,axis=1)[new_exposures.columns[-1]]
	print("\nCumulative reporting rate:")
	cum_rr = cum_rr_samples.describe(percentiles=[0.025,0.25,0.5,0.75,0.975]) 
	print(cum_rr)

	## Sample mortality
	ifr_samples = np.random.normal(ifr["mean"].values,
								   np.sqrt(ifr["var"].values),
								   size=((num_samples,len(ifr))))/100.
	ifr_samples = np.clip(ifr_samples,0,None)
	delay_samples = np.exp(np.random.normal(2.8329,0.42/4.,size=(num_samples,)))
	delay_samples = np.clip(delay_samples,None,new_exposures.shape[1]-1)
	destined_deaths, _ = SampleOutcome(new_exposures.values,
									   ifr_samples,
									   np.round(delay_samples).astype(int))
	deaths_occured = np.dot(destined_deaths,mP.T)

	## Then sample hosps
	ihr_samples = np.random.normal(ihr["mean"].values,
								   np.sqrt(ihr["var"].values),
								   size=((num_samples,len(ifr))))/100.
	delay_samples = np.random.normal(model.D_e+2.1,2.65/4.,size=(num_samples,))
	delay_samples = np.clip(delay_samples,0,new_exposures.shape[1]-1)
	destined_hospital, _ = SampleOutcome(new_exposures.values,
										 ihr_samples,
										 np.round(delay_samples).astype(int))
	hospital_occured = np.dot(destined_hospital,hP.T)

	## Assess the fit
	print("\nFit quality...")
	print("For hospitalizations:")
	fit_quality(hospital_occured,"hosp")
	print("For deaths:")
	fit_quality(deaths_occured,"deaths")
	print("For cases:")
	fit_quality(case_samples,"cases")

	## Compute the rolling weekly reporting rate
	i_samples = pd.DataFrame(samples[:,2,:].T,
							 index=plot_time).loc[cases.index[0]:cases.index[-1]].T
	weekly_reporting_samples = (cases.values)/(i_samples)
	weekly_reporting_samples = 100*(1.-(1. - weekly_reporting_samples.T.rolling(7).mean())**model.D_i)
	weekly_reporting_samples = weekly_reporting_samples.loc["2020-03-01":].T
	l0, h0, l1, h1, l2, h2 = low_mid_high(weekly_reporting_samples.values)
	avg_rr = 100*(1 - (1.-rr.rolling(7).mean())**model.D_i).loc[weekly_reporting_samples.columns]
	var_rr = weekly_reporting_samples.var(axis=0)
	rr_df = pd.DataFrame(np.array([l0, h0, l1, h1, l2, h2, avg_rr,var_rr]).T,
						 index=weekly_reporting_samples.columns,
						 columns=["l0", "h0", "l1", "h1", "l2", "h2", "avg_rr","var_rr"])
	rr_df = rr_df.iloc[:-1]
	rr_df = rr_df.reindex(cases.index).fillna(method="bfill").fillna(method="ffill")
	rr_df.to_pickle("pickle_jar\\model_rep_rate.pkl")

	## If specified, output a dataframe with all
	## the key model pieces (beta, sig_eps, importations, severity
	## rates) needed to make cross-model comparisons, forecasts, etc.
	if hp["model_pickle"] is not None:
		model_pickle = pd.concat([beta_t.rename("beta_t"),
								  sig_eps.rename("sig_eps"),
								  importations.rename("z_t"),
								  rr.rename("rr")],axis=1)
		ifr.columns = ["ifr_"+c for c in ifr.columns]
		ihr.columns = ["ihr_"+c for c in ihr.columns]
		raw_vax.columns = ["vax_"+c for c in raw_vax.columns]
		ihr_sf_series = pd.Series(len(model_pickle)*[ihr_scale_factor],
								  index=model_pickle.index,
								  name="ihr_scale_factor")
		pulse_exp_series = pd.Series([pulse[1]]+(len(model_pickle)-1)*[0],
								  index=model_pickle.index,
								  name="pulse")
		pulse_std_series = pd.Series([pulse_std[1]]+(len(model_pickle)-1)*[0],
								  index=model_pickle.index,
								  name="pulse_std")
		model_pickle = pd.concat([model_pickle,
								  ifr[["ifr_mean","ifr_var"]],
								  ihr[["ihr_mean","ihr_var"]],
								  ihr_sf_series,
								  pulse_exp_series,
								  pulse_std_series,
								  raw_vax],axis=1)
		model_pickle = model_pickle.loc[:hp["plot_time_end"]]
		model_pickle.to_pickle("pickle_jar\\"+hp["model_pickle"])
		print("\nSerialized model dataframe = ")
		print(model_pickle)
		print(model_pickle[["ihr_mean","ihr_var"]].mean())
		print(ihr_scale_factor*ihr_table)
		#sys.exit()

	########################################################################################
	#### Plotting.
	###########

	## One big figure demonstrating fits and key outputs.
	fig, axes = plt.subplots(3,2,sharex=True,figsize=(6.5*(4/2),10))
	axes = axes.T
	ax_pos = {"cases":(0,0),"hosps":(0,1),"deaths":(0,2),
			  "beta":(1,0),"rr":(1,2),"prev":(1,1)}
	colors = {"cases":"#FFBB00","hosps":"#FB6542","deaths":"#375E97",
			  "prev":"#E71D36","rr":"#3F681C","beta":"#791E94",
			  "ifr":"#f100e5","immunity":"#FFBC42","inc":"#353866"}
	
	## Overall set up
	for row in axes:
		for ax in row:
			axes_setup(ax)
			ax.grid(color="grey",alpha=0.2)
	mpticks = pd.date_range(hp["tr_start"],"2021-04-01",freq="MS")[::3]
	mptick_labels = [t.strftime("%b")+(t.month == 3)*"\n{}".format(t.year) for t in mpticks]
	T = (mpticks[0]-plot_time[0]).days

	## Cases
	l0, h0, l1, h1, l2, h2 = low_mid_high(case_samples)
	avg = case_samples.mean(axis=0)
	i,j = ax_pos["cases"]
	c = colors["cases"]
	axes[i,j].fill_between(plot_time[T:],l1[T:],h1[T:],facecolor=c,edgecolor="None",alpha=0.3,zorder=2)
	#axes[i,j].fill_between(plot_time[T:],l2[T:],h2[T:],facecolor=c,edgecolor="None",alpha=0.6,zorder=3)
	axes[i,j].plot(plot_time[T:],avg[T:],lw=2,color=c)
	axes[i,j].plot(raw_timeseries.loc[mpticks[0]:plot_time[-1],"cases"],
			  marker=".",ls="None",color="k",markersize=8,zorder=4)
	axes[i,j].set_ylabel("Positive tests",color=c)
	axes[i,j].set_ylim((0,None))
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)

	## Hosps
	l0, h0, l1, h1, l2, h2 = low_mid_high(hospital_occured)
	avg = hospital_occured.mean(axis=0)
	i,j = ax_pos["hosps"]
	c = colors["hosps"]
	axes[i,j].fill_between(plot_time[T:],l1[T:],h1[T:],facecolor=c,edgecolor="None",alpha=0.3)
	#axes[i,j].fill_between(plot_time[T:],l2[T:],h2[T:],facecolor=c,edgecolor="None",alpha=0.6)
	axes[i,j].plot(plot_time[T:],avg[T:],lw=2,color=c)
	axes[i,j].plot(raw_timeseries.loc[mpticks[0]:plot_time[-1],"hosp"],
			  marker=".",ls="None",color="k",markersize=8,zorder=4)
	axes[i,j].set_ylabel("Hosp. admissions",color=c)
	axes[i,j].set_ylim((0,None))
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)

	## Deaths
	l0, h0, l1, h1, l2, h2 = low_mid_high(deaths_occured)
	avg = deaths_occured.mean(axis=0)
	i,j = ax_pos["deaths"]
	c = colors["deaths"]
	axes[i,j].fill_between(plot_time[T:],l1[T:],h1[T:],facecolor=c,edgecolor="None",alpha=0.3)
	#axes[i,j].fill_between(plot_time[T+1:],l2[T:],h2[T:],facecolor=c,edgecolor="None",alpha=0.6)
	axes[i,j].plot(plot_time[T:],avg[T:],lw=2,color=c)
	axes[i,j].plot(raw_timeseries.loc[mpticks[0]:plot_time[-1],"deaths"],
			  marker=".",ls="None",color="k",markersize=8,zorder=4)
	axes[i,j].set_ylabel("Daily deaths",color=c)
	axes[i,j].set_ylim((0,None))
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)

	## Prevalence
	l0, h0, l1, h1, l2, h2 = low_mid_high(100*prevalence.values)
	avg = (100*prevalence.values).mean(axis=0)
	i,j = ax_pos["prev"]
	c = colors["prev"]
	axes[i,j].fill_between(plot_time[T:],l1[T:],h1[T:],facecolor=c,edgecolor="None",alpha=0.3,zorder=5)
	#axes[i,j].fill_between(plot_time[T:],l2[T:],h2[T:],facecolor=c,edgecolor="None",alpha=0.8)
	axes[i,j].plot(plot_time[T:],avg[T:],color=c,lw=2,zorder=6)
	ylim = axes[i,j].get_ylim()
	scan_result = GetSCANPrevalence("published")
	for t in scan_result:
		d1, d2, m, l, h = t
		d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)
		axes[i,j].fill_between([d1,d2],[l,l],[h,h],alpha=0.3,facecolor="grey",edgecolor="None",zorder=1)
		axes[i,j].plot([d1,d2],[m,m],lw=1,color="k",zorder=2)
	scan_result = GetSCANPrevalence("recent")
	for t in scan_result:
		d1, d2, m, l, h = t
		d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)
		axes[i,j].fill_between([d1,d2],[l,l],[h,h],alpha=0.3,facecolor="grey",edgecolor="None",zorder=1)
		axes[i,j].plot([d1,d2],[m,m],lw=1,color="k",zorder=2)
	axes[i,j].plot([],c="k",lw=1,label="SCAN reports")
	scan_result = GetSCANPrevalence("lancet")
	for t in scan_result:
		d1, d2, m, l, h = t
		d1, d2 = pd.to_datetime(d1), pd.to_datetime(d2)
		axes[i,j].fill_between([d1,d2],[l,l],[h,h],alpha=0.3,facecolor="#DDBC95",edgecolor="None",zorder=1)
		axes[i,j].plot([d1,d2],[m,m],lw=1,color="#B38867",zorder=2)
	axes[i,j].plot([],c="#B38867",lw=1,label=r"Du et al estimate")
	axes[i,j].set_ylabel(r"Pop. prevalence (%)",color=c)
	axes[i,j].set_ylim((0,ylim[1]))
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)
	leg = axes[i,j].legend(loc=4,frameon=False,fontsize=14)
	leg.set_zorder(8)

	## Reporting rate
	rr_df = rr_df.loc[weekly_reporting_samples.columns[0]:]
	i,j = ax_pos["rr"]
	c = colors["rr"]
	#axes[i,j].fill_between(rr_df.index,rr_df["l1"].values,rr_df["h1"].values,facecolor=c,edgecolor="None",alpha=0.3)
	axes[i,j].fill_between(rr_df.index,
						   avg_rr - 2.*np.sqrt(var_rr),
						   avg_rr + 2.*np.sqrt(var_rr),
						   facecolor=c,edgecolor="None",alpha=0.3)
	#axes[i,j].fill_between(rr_df.index,rr_df["l2"].values,rr_df["h2"].values,facecolor=c,edgecolor="None",alpha=0.6)
	axes[i,j].plot(rr_df.index,rr_df["avg_rr"].values,color=c,lw=2)
	weekly_tests = (timeseries["cases"]+timeseries["negatives"]).resample("W-SUN").sum()
	weekly_tests = weekly_tests.reindex(rr_df.index).fillna(method="bfill").dropna()
	axes[i,j].plot((weekly_tests/(weekly_tests.max()))*(avg_rr + 2.*np.sqrt(var_rr)).max()*1.05,c="k",lw=2,ls="dashed",
				   label="Relative testing volume")
	axes[i,j].set_ylabel("Reporting rate (%)",color=c)
	axes[i,j].set_ylim((0,None))
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)
	axes[i,j].legend(loc=4,frameon=False,fontsize=14)

	## Transmission rate
	i,j = ax_pos["beta"]
	c = colors["beta"]
	axes[i,j].fill_between(r0_estimates.index,
					  r0_estimates["r0_t"].values-2.*r0_estimates["std_err"].values,
					  r0_estimates["r0_t"].values+2.*r0_estimates["std_err"].values,
					  facecolor=c,edgecolor="None",alpha=0.3,zorder=3)
	axes[i,j].plot(r0_estimates["r0_t"],lw=2,color=c,zorder=4)
	rtlive = GetRtLive().loc[r0_estimates.index[0]:]
	axes[i,j].fill_between(rtlive.index,
						   rtlive["lower_80"].values,rtlive["upper_80"].values,
						   facecolor="grey",edgecolor="None",alpha=0.3)
	axes[i,j].plot(rtlive["mean"],c="k",lw=2,ls="dashed",
				   label="rt.live estimate")
	#axes[i,j].plot(rtlive["lower_80"],c="k",lw=1,ls="dashed")
	#axes[i,j].plot(rtlive["upper_80"],c="k",lw=1,ls="dashed")
	#axes[i,j].axhline(1,ls="dashed",lw=2,c="grey",zorder=2)
	axes[i,j].set_ylabel(r"R$_e$",color=c)
	axes[i,j].set_ylim((0,None))
	axes[i,j].set_yticks([0,1,2,3])
	axes[i,j].set_xticks(mpticks)
	axes[i,j].set_xticklabels(mptick_labels)
	axes[i,j].legend(loc=4,frameon=False,fontsize=14)

	## Done
	fig.tight_layout()
	fig.savefig("_plots\\model_overview.png")
	plt.show()
	