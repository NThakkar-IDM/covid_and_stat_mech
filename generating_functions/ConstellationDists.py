""" ConstellationDists.py

Visualization of the convergence to the mean, non-interacting degree distribution
in the more general contellation distributions. Figure 2 in the generating function
paper. """
import sys
sys.path.append("..\\")
import methods

## standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For probability calculations
from scipy.special import gamma, gammaln, binom

## For sampling
from methods.forest_sampling import random_rewiring_sampler

## For reproducability
plt.rcParams["font.size"] = 22.
np.random.seed(2)

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def lnf(x,k,I,N):
	ans = gammaln(k*(I-x)+N)+gammaln(k*I)\
		-gammaln(k*I+N)-gammaln(k*(I-x))
	return ans

def f(x,k,I,N):
	ans = lnf(x,k,I,N)
	return np.exp(ans)

def lnfr(x,r,k,I,N):
	ans = gammaln(k*(I-x)+N-x*r)+gammaln(k*I)+gammaln(N+1)\
		-gammaln(k*I+N)-gammaln(k*(I-x))-gammaln(N-x*r+1)\
		+x*gammaln(k+r)-x*gammaln(r+1)-x*gammaln(k)
	return ans

def fr(x,r,k,I,N):
	ans = lnfr(x,r,k,I,N)
	return np.exp(ans)

def gamma_poisson_density(x,m,k):
	ln_N = gammaln(x+k)-gammaln(x+1)-gammaln(k)
	ln_sf = x*np.log(m/(m+k))+k*np.log(k/(m+k))
	return np.exp(ln_N+ln_sf)

if __name__ == "__main__":

	## Set up the examples
	Is = np.arange(8,901)
	k = 0.015037 
	mu = 0.266542
	Ns = (mu*Is).astype(np.int64)
	p = mu/(mu+k)

	## Set up the individual-level distributions
	ls = np.arange(max(Ns)+5)
	p_l = pd.Series(gamma_poisson_density(ls,mu,k),
					index=ls,name="p_l")
	p_l = p_l/(p_l.sum())

	## Compute the asymptotic distribution via Section 4 in the
	## paper.
	fk = k*k + 0.5*k + (1./12.)
	pr_C_asy = p_l[:,None]*(np.ones(Is.shape)[None,:])
	pr_C_asy *= (1.-fk*(mu/(k+mu))/(k*Is[None,:]))
	pr_C_asy *= (1.+ls[:,None]/((k+mu)*Is[None,:]))**k
	pr_C_asy *= (1.-(k*(ls[:,None]**2)-k*ls[:,None]-2*mu*ls[:,None])/(2*mu*(k+mu)*Is[None,:]))
	pr_C_asy = pr_C_asy/(pr_C_asy.sum(axis=0)[None,:])

	## And the asymptotic std error
	var_r_asy = np.exp(gammaln(k+ls)-gammaln(ls+1)\
						-gammaln(k)+k*np.log(k/(mu+k)))
	var_r_asy = (var_r_asy[:,None])*(1. - var_r_asy[:,None])/(Is[None,:])
	std_r_asy = np.sqrt(var_r_asy)

	## Construct samples
	sample_Is = np.random.choice(Is,replace=True,size=(1000,))
	sample_Ns = (mu*sample_Is).astype(np.int64)

	## And sample the associated graphs
	samples = []
	for N, I in zip(sample_Ns,sample_Is):
		
		## Set up the distributions
		p_l_given_N = p_l.loc[:N]/(1. - (p_l.loc[N+1:].sum()))
		non_zero_p_l = p_l_given_N.values[1:]/(1.-p_l_given_N.loc[0])
		
		## sample a graph
		sample = random_rewiring_sampler(N,I,
										 p_l_given_N.index,
										 non_zero_p_l,
										 num_samples=1,
										 )[0]
		
		## store it
		samples.append(sample)

	## Put it together in a data frame
	df = pd.DataFrame(np.array([sample_Is,sample_Ns]).T,
					  columns=["I","N"])
	df["const"] = samples

	## Compute some stats
	df["0s"] = df["const"].apply(lambda a: (a == 0).sum())
	df["1s"] = df["const"].apply(lambda a: (a == 1).sum())
	df["2s"] = df["const"].apply(lambda a: (a == 2).sum())
	df["3+"] = df["const"].apply(lambda a: (a >= 3).sum())

	## Compute the analytic probability distribution
	Z = binom(k*Is+Ns-1,Ns)
	ln_const_dist = np.log(binom(k+ls[:,None]-1,ls[:,None]))\
					+np.log(binom(k*(Is[None,:]-1)+Ns[None,:]-ls[:,None]-1,
								  Ns[None,:]-ls[:,None]))\
					-np.log(Z)
	pr_C = np.exp(ln_const_dist)

	## Compute the estimates
	pr_0 = pr_C_asy[0,:]
	pr_1 = pr_C_asy[1,:]
	pr_2 = pr_C_asy[2,:]
	pr_3_plus = pr_C_asy[3:,:].sum(axis=0)

	## Compute the variance in the distribution
	var_r = Is[None,:]*Is[None,:]*(fr(2,ls[:,None],k,Is[None,:],Ns[None,:])-\
								   fr(1,ls[:,None],k,Is[None,:],Ns[None,:])**2)\
			+Is[None,:]*(fr(1,ls[:,None],k,Is[None,:],Ns[None,:])-\
						 fr(2,ls[:,None],k,Is[None,:],Ns[None,:]))
	std_r = np.sqrt(var_r)/Is[None,:]
	
	## Compute the appropriate std errors
	std_0 = std_r_asy[0,:]
	std_1 = std_r_asy[1,:]
	std_2 = std_r_asy[2,:]
	std_3_plus = std_r_asy[3,:]

	## Plots
	fig, axes = plt.subplots(2,2,
							 sharex=True,
							 figsize=(12,8))
	axes = axes.reshape(-1)
	for ax in axes:
		axes_setup(ax)
		ax.grid(color="grey",alpha=0.2)

	## Plot the samples
	axes[0].plot(df["I"],df["0s"]/df["I"],
				marker=".",markersize=8,
				markeredgecolor="None",
				ls="None",
				color="k",
				alpha=0.15,label="Samples")
	axes[1].plot(df["I"],df["1s"]/df["I"],
				marker=".",markersize=8,
				markeredgecolor="None",
				ls="None",
				color="k",
				alpha=0.15,label="Samples")
	axes[2].plot(df["I"],df["2s"]/df["I"],
				marker=".",markersize=8,
				markeredgecolor="None",
				ls="None",
				color="k",
				alpha=0.15)
	axes[3].plot(df["I"],df["3+"]/df["I"],
				marker=".",markersize=8,
				markeredgecolor="None",
				ls="None",
				color="k",
				alpha=0.15)

	## Plot the estimates
	## 0
	axes[0].plot(Is,pr_0,lw=3,color="#3079F7",label="Asymptotic distribution")
	axes[0].plot(Is,pr_0-2*std_0,lw=2,color="#3079F7")
	axes[0].plot(Is,pr_0+2*std_0,lw=2,color="#3079F7")
	
	## 1
	axes[1].plot(Is,pr_1,lw=3,color="#3079F7",label="Asymptotic distribution")
	axes[1].plot(Is,pr_1-2*std_1,lw=2,color="#3079F7")
	axes[1].plot(Is,pr_1+2*std_1,lw=2,color="#3079F7")
	
	## 2
	axes[2].plot(Is,pr_2,lw=3,color="#3079F7")
	axes[2].plot(Is,pr_2-2*std_2,lw=2,color="#3079F7")
	axes[2].plot(Is,pr_2+2*std_2,lw=2,color="#3079F7")
	
	## 3
	axes[3].plot(Is,pr_3_plus,lw=3,color="#3079F7")
	axes[3].plot(Is,pr_3_plus-2*std_3_plus,lw=2,color="#3079F7")
	axes[3].plot(Is,pr_3_plus+2*std_3_plus,lw=2,color="#3079F7")
	
	## Add the non-interacting lines
	axes[0].axhline(p_l.loc[0],color="#F73079",lw=3,ls="dashed",label="Noninteracting limit")
	axes[1].axhline(p_l.loc[1],color="#F73079",lw=3,ls="dashed",label="Noninteracting limit")
	axes[2].axhline(p_l.loc[2],color="#F73079",lw=3,ls="dashed")
	axes[3].axhline(p_l.loc[3:].sum(),color="#F73079",lw=3,ls="dashed")

	## Details
	axes[0].set_ylabel(r"Pr$(\ell = 0)$")
	axes[1].set_ylabel(r"Pr$(\ell = 1)$")
	axes[2].set_ylabel(r"Pr$(\ell = 2)$")
	axes[3].set_ylabel(r"Pr$(\ell \geq 3)$")
	axes[2].set_xlabel("Infectious population")
	axes[3].set_xlabel("Infectious population")
	
	## Limits?
	for ax in axes:
		ax.set_xlim((0,Is[-1]))
	for ax in axes[1:]:
		ax.set_ylim((-0.005,None))
	axes[0].set_ylim((None,1))

	## Set up a legend
	legend = axes[1].legend(loc=1,frameon=True,fontsize=18)
	legend.get_frame().set_linewidth(0.0)
	legend.get_frame().set_alpha(0.6)

	## Finish up
	fig.tight_layout()
	fig.savefig("..\\_plots\\constellation_distributions.png")
	plt.show()

