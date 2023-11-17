""" forest_sampling.py

Functions to support random graph sampling and transmission tree
construction. """
import sys
import numpy as np

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