'''
Created on 13 Mar 2013

@author: root
'''

#set of functions for multinomial observations in HMM

from numpy import *
from hmm.util import arrayWithOne
import sys

#use mode of multinomial to calculate ML estimate of O given obs and distribution over latent states:
def optOMult(xs,Z0,NW):
    #MAP approach (with prior pseudo count 0.5
    alpha = 0.5
    N = len(xs)
    (N1,K) = shape(Z0)
    assert N==N1
    #convert xs to matrix X:
    X0 = zeros((N,NW))
    for n in range(N): X0[n,:] = arrayWithOne(xs[n],NW)
    #for each word w in NW, calcualate frequency
    F = zeros( (NW,K) ) + alpha
    for k in range(K):
        Zsum = Z0[:,k].sum()
        for w in range(NW): 
            F[w,k] += (X0[:,w]*Z0[:,k]).sum()
            F[w,k] /= (Zsum+NW*alpha)
    #print 'F',F
    #normalise:
    #O = F/reshape(F.sum(axis=1),(K,1))
    
    #make sure no zero entries (otherwise we get into trouble with testing when an obs with zero probability happens)
    assert all(F>0),F.T
    assert all((F.sum(axis=0)-1.0) < 0.000001),F.T
    
    return F.T


def logOptOMult(xs,Z0,NW):
    #MAP approach (with prior pseudo count 0.5)
    alpha = 0.0 # 0.5
    N = len(xs)
    (N1,K) = shape(Z0)
    assert N==N1,'%i %i'%(N,N1)
    #convert xs to matrix X:
    X0 = zeros((N,NW))
    for n in range(N): X0[n,:] = arrayWithOne(xs[n],NW)
    #for each word w in NW, calcualate frequency
    lgF = log(zeros( (NW,K) ) + alpha)
    for k in range(K):
        Zsum = Z0[:,k].sum()
        if Zsum>0:
            for w in range(NW):
                log_xz_sum = -inf
                for n in range(N):
                    log_xz_sum = logaddexp(log_xz_sum, log(X0[n,w]) + log(Z0[n,k]))
                lgF[w,k] = log_xz_sum-log(Zsum + NW*alpha)

    #make sure normalized:
    assert all( abs(exp(lgF).sum(axis=0)-1.) < 1e-4), exp(lgF).sum(axis=0)
    #make sure no zero entries (otherwise we get into trouble with testing when an obs with zero probability happens)
    #assert all(lgF>-inf),lgF.T
    
    return lgF.T


def margForOMult(z,O):
    #marginalise over discrete latent states for multinomial observations:
    (K,NW)=shape(O)
    (K1,)=shape(z)
    assert K==K1
    pr_obs = zeros(NW)
    for k in range(K): pr_obs += z[k]*O[k,:]
    return pr_obs

def logMargForOMult(z,O):
    #marginalise over discrete latent states for multinomial observations:
    (K,NW)=shape(O)
    (K1,)=shape(z)
    assert K==K1
    log_pr_obs = -inf + zeros(NW)
    for k in range(K):
        for w in range(NW):
            assert O[k,w]>0.,'k=%i,w=%i'%(k,w)
            log_pr_obs[w] = logaddexp(log_pr_obs[w], z[k]+log(O[k,w]))
    return log_pr_obs