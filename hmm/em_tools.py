'''
Created on Mar 25, 2013

@author: James McInerney
'''

from random import choice
from numpy import *
from scipy.stats import norm
import sys

seterr(all='raise')

def gen(N,L,DW=24,EXTRA_VERBOSE=False):
    #loc, day of week, hour of day, departure from routine:
    X,D,H,E = zeros((N,L)), zeros((N,DW)), zeros(N), zeros(N)
    #priors:
    alpha,beta,delta,gamma = 0.2,1,2,2
    mu_mu, mu_sigma = 12., 5.
    sigma_conc = 5.0
    #generative model:
    #step 1. draw parmeters from priors
    T = array([random.dirichlet(alpha+zeros(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    R = random.beta(delta,gamma) #probability of departure from routine
    Q = array([[random.dirichlet(beta+zeros(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    S_mu = array([[random.normal(mu_mu,mu_sigma) for _ in range(L)] for _ in range(2)])
    S_sigma = array([[random.gamma(sigma_conc) for _ in range(L)] for _ in range(2)])
    
    #step 2. draw observations
    for n in range(N):
        E[n] = random.binomial(1,R)
        if E[n]==0:
            if EXTRA_VERBOSE: print 'e_0',E[n]
            prev_loc = X[n-1,:].argmax()
            pr_x = T[prev_loc,:]
        else:
            if EXTRA_VERBOSE: print 'e_1',E[n]
            #uniform random choice:
            pr_x = ones(L)/float(L)
        X[n,:] = random.multinomial(1, pr_x)
        l_n = X[n,:].argmax()
        #finally, draw time context using location and departure random variable instantiations:
        if E[n]==0:
            #then day/time is drawn from parameters associated with location:
            pr_h = Q[0,l_n,:]
            h_mu,h_sigma = S_mu[0,l_n], S_sigma[0,l_n] 
        else:
            #then day/time is drawn from parameters associated with departure from routine:
            #(N.B., Q[1,i,:] is not used for i!=0
            pr_h = Q[1,0,:]
            h_mu,h_sigma = S_mu[1,0], S_sigma[1,0] 
        D[n,:] = random.multinomial(1, pr_h)
        H[n] = random.normal(h_mu,h_sigma)
    
    return X,D,H,E,S_mu,S_sigma,Q,R,T



def optQ(D,E,X,DW,beta):
    #consider conditional counts for time obs:
    (N,L)=shape(X)
    K=2
    Q = zeros((K,L,DW)) + beta
    for n in range(N):
        l_n = X[n,:].argmax()
        Q[0,l_n,:] += D[n,:]*(1-E[n])
        Q[1,0,:] += D[n,:]*(E[n])
    #normalise by dimension 2
    Q = Q / reshape(Q.sum(axis=2), (K,L,1))
    return Q
        


def optR(E,delta,gamma):
    #find MAP estimate of the departure from routine probability (parameter)
    #easy: just consider the values of e_n for all n in N
    E_expl = E_explicit(E)
    F = E_expl.sum(axis=0) + array([gamma,delta]) #gamma: controls 0, delta: controls 1 (OPPOSITE WAY ROUND TO beta gen function)
    R = F / F.sum()
    return R[1]




def optT(X,E,Mu_prior,l_init=0,VERBOSE=False):
    #MAP estimate of location transition matrix
    
    (N,L) = shape(X)
    FT = Mu_prior #zeros((L,L)) + alpha
    
    for n in range(N):
        l_next = X[n,:].argmax()
        if n>0: l_prev = X[n-1,:].argmax()
        else: l_prev = l_init
        FT[l_prev,l_next]+= (1.-E[n])
    
    #normalise counts over rows of transition matrix:
    T = FT / reshape(FT.sum(axis=1), (L,1))
    if VERBOSE: print 'opt T',T
    return T



#maximum likelihood estimator of temporal component parameters:
def optS_ML(H,E,X,MAX_S_SIGMA = 0.000001):
    K=2
    (N,L)=shape(X)
    S_mu, S_sigma = zeros((K,L)), zeros((K,L))
    F,Ns = zeros((K,L)),zeros((K,L))
    for n in range(N):
        l_n = X[n,:].argmax()
        F[0,l_n] += (1.-E[n])*H[n]
        Ns[0,l_n] += 1.-E[n]
        F[1,0] += E[n]*H[n]
        Ns[1,0] += E[n]
    for l in range(L):
        if Ns[0,l]>0: S_mu[0,l] = F[0,l]/Ns[0,l]
        S_mu[1,l] = F[1,0]/Ns[1,0]
    Fsq = zeros((K,L))
    for n in range(N):
        l_n = X[n,:].argmax()
        e_n0,e_n1 = 1.-E[n],E[n]
        Fsq[0,l_n] += (e_n0*(S_mu[0,l_n]-H[n]))**2
        Fsq[1,0] += (e_n1*(S_mu[1,0]-H[n]))**2
    for l in range(L):
        if Ns[0,l]>0: S_sigma[0,l] = max(sqrt(Fsq[0,l]/Ns[0,l]),MAX_S_SIGMA)
        S_sigma[1,0] = max(sqrt(Fsq[1,0]/Ns[1,0]), MAX_S_SIGMA)
    
    print 'Ns',Ns
    return S_mu,S_sigma
    
#old version:
def optS1(H,E,X,MAX_S_SIGMA = 0.01):
    #find ML (FOR NOW) esimtate of parameter controlling hour of day:
    K=2
    (N,L)=shape(X)
    S_mu, S_sigma = zeros((K,L)), zeros((K,L))
    E_expl = E_explicit(E) #col0: pr(e_n) = 0, col1: pr(e_n) = 1
    E_sum = E_expl.sum(axis=0)
    H1 = reshape(H,(N,1))
    F = H1*E_expl
    Hsum = F.sum(axis=0)
    #find conditional sum (given loc):
    Hcondsum, Econdsum = zeros((K,L)), zeros(L)
    for n in range(N):
        l_n = X[n,:].argmax()
        Hcondsum[:,l_n] += E_expl[n,:]*H[n]
        Econdsum[l_n] += E_expl[n,1] 
    S_mu[0,:] = Hcondsum[0,:]/Econdsum #ML estimate
    S_mu[1,0] = Hsum[1]/E_sum[1]
    #now calculate variance for H when e_n = 0
    Sigmacondsum = zeros((K,L))
    for n in range(N):
        l_n = X[n,:].argmax()
        Sigmacondsum[:,l_n] += (E_expl[n,:]*H[n] - S_mu[:,l_n])**2
    S_sigma[0,:] = map(lambda x: max(x,MAX_S_SIGMA), sqrt(Sigmacondsum[0,:]/Econdsum))
    S_sigma[1,0] = max(sqrt(Sigmacondsum[1,:].sum()/E_sum[1]), MAX_S_SIGMA)
#    S_sigma[1,0] = sqrt(((F - S_mu[1,0])**2).sum(axis=0)/E_sum)
#    print 'S_sigma',S_sigma
#    sys.exit(0)
    return S_mu,S_sigma

def calcLogLik(E,X,D,H,T,Q,R,S_mu,S_sigma,useS,l_init=0):
    #calc total logLik of model given parameters and latent variables:
    totalLogLik = 0.0
    (N,L)=shape(X)
    for n in range(N):
        #calculate observations for this time step:
        l_next = X[n,:].argmax()
        if n>0: l_prev = X[n-1,:].argmax()
        else: l_prev = l_init
        d_n = D[n,:].argmax()
        if useS: h_n = H[n]
        e_n0, e_n1 = 1.-E[n], E[n]
        #print 'S_mu,S_sigma',S_mu[0,l_next],S_sigma[0,l_next]
        #print 'vals0',T[l_prev,l_next], Q[0,l_next,d_n], lognormpdf(h_n,S_mu[0,l_next],S_sigma[0,l_next]), 1.-R
        #print 'vals1',S_mu[1,0], S_sigma[1,0], Q[1,0,d_n], norm.pdf(h_n,S_mu[1,0],S_sigma[1,0]), lognormpdf(h_n,S_mu[1,0],S_sigma[1,0]), R
        logLik0 = e_n0*(log(T[l_prev,l_next]) + \
                    log(Q[0,l_next,d_n]) + \
                            + log(1.-R))
        if useS: logLik0 += e_n0*lognormpdf(h_n,S_mu[0,l_next],S_sigma[0,l_next])
        #remember: location is not used for temp observations when e_n = 1
        logLik1 = e_n1*(-log(L) + log(Q[1,0,d_n]) + log(R))
        if useS: logLik1 += e_n1*lognormpdf(h_n,S_mu[1,0],S_sigma[1,0])
        
        #total:
        #print 'H[n],S_mu[0],S_sigma[0]',H[n],S_mu[0],S_sigma[0]
        totalLogLik += logLik0 + logLik1
        assert not(isnan(totalLogLik))
    return totalLogLik

def lowerBoundLL(N,L,DW,H):
    #calculates lower estimate of what log lik can be, based on number of data points, number of locations, etc.
    logLik = N*(-log(L) - log(DW) - log(2)) + lognormpdf(H,12.,20.).sum()
    #print 'lognorm H',lognormpdf(H,12.,20.)
    return logLik

def run_em(X,D,H,priors,useS=False,S_grnd=None,Q_grnd=None,E_grnd=None,final=True,doPlot=0,max_iter=100,showLowerBoundLL=False,EXTRA_VERBOSE=False):
    (Mu_prior,beta,delta,gamma,mu_mu,mu_sigma,sigma_conc) = priors
    (N,L) = shape(X)
    (N1,DW) = shape(D)
    if H is not None: 
        (N2,)=shape(H)
        assert N==N2
    assert N==N1
    
    fixS,fixQ,fixE = False,False,False
    #step 1. draw parmeters from priors
    if E_grnd is None:
        E = array([choice([0.,1.]) for _ in range(N)])
    else:
        E = E_grnd
        fixE = True
    T = array([random.dirichlet(ones(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    R = random.beta(1,1) #probability of departure from routine
    if Q_grnd is None:
        Q = array([[random.dirichlet(ones(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    else:
        Q = Q_grnd
        fixQ = True
    if S_grnd is None:
        S_mu = array([[random.normal(mu_mu,mu_sigma) for _ in range(L)] for _ in range(2)])
        S_sigma = array([[random.gamma(sigma_conc) for _ in range(L)] for _ in range(2)])
    else:
        S_mu,S_sigma = S_grnd
        fixS = True
    
    itr = 0
    logLikDiff, minLogLikDiff, logLikPrev = 100, 0.05, -9999999999999999.
    logLik = [] #logLiks of each iteration
    while itr<max_iter and logLikDiff > minLogLikDiff:
        print 'iteration %i...'%itr
        #E-step:
        # p(l_n | l_n-1, e_n) p(d_n | l_n, e_n)
        # = p(l_n, d_n | l_n-1, e_n) p(e_n | R)
        # propto p(e_n | l_n, d_n, l_n-1)
        if not fixE: E = estep(X,D,H,T,Q,R,S_mu,S_sigma,useS)
        
        #M-step:
        R = optR(E,delta,gamma)
        T = optT(X,E,Mu_prior,VERBOSE=EXTRA_VERBOSE)
        if not fixQ: Q = optQ(D,E,X,DW,beta)
        if useS and (not fixS): S_mu, S_sigma = optS(H,E,X)

        logLik.append(calcLogLik(E,X,D,H,T,Q,R,S_mu,S_sigma,useS))
        if EXTRA_VERBOSE:
            print 'S_mu',S_mu
            print 'S_sigma',S_sigma
            print 'E',E
        print 'totalLogLik',logLik
        
        #loop management:
        logLikDiff = abs(logLik[-1] - logLikPrev)
        logLikPrev = logLik[-1]
        itr += 1
    
    if itr==max_iter: print 'REACHED MAX ITERATIONS'
    if doPlot: plot(range(itr),logLik)
    if showLowerBoundLL:
        lb = lowerBoundLL(N,L,DW,H)
        plot(range(itr),[lb for _ in range(itr)],linestyle='--')
    if doPlot and final:
        show()
    elif doPlot and not(final):
        figure()

    return E,T,Q,R,S_mu,S_sigma,logLik[-1]

def calcRMSE(E0,E1):
    #finds the error between two vectors:
    (N,)=shape(E0)
    (N1,)=shape(E1)
    assert N==N1
    Dif = (E0 - E1)**2
    return sqrt(Dif.sum()/N)


def calcME(E0,E1):
    #finds the error between two vectors:
    (N,)=shape(E0)
    (N1,)=shape(E1)
    assert N==N1
    return abs(E0-E1).sum()/N


    

def makeCoarse(D,crs):
    (N,NHW)=shape(D)
    D_crs = zeros((N,1+NHW/crs))
    for n in range(N):
        d_n = int(D[n,:].argmax()/crs)
        #print 'd_n',d_n
        #print 'D_crs[n,:] (%i)'%len(D_crs[n,:]),D_crs[n,:]
        D_crs[n,d_n]=1.
    return D_crs


