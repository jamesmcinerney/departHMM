'''
Created on 11 Nov 2012

@author: James McInerney
'''



from numpy import *


seterr(all='raise')

def gen(N,L,DW=168,EXTRA_VERBOSE=False):
    #N: number of observations
    #L: number of discrete locations
    #DW: number of time slots per period (e.g., 1 week = 168 hours)
    
    #observations/latent variables: loc, day of week, departure from routine:
    X,D,E = zeros((N,L)), zeros((N,DW)), zeros(N)
    #priors:
    alpha,beta,delta,gamma = 1.0,1.0,1.0,1.0
    #generative model:
    #step 1. draw parmeters from priors
    mu = array([random.dirichlet(alpha+zeros(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    r = random.beta(delta,gamma) #probability of departure from routine
    w = array([[random.dirichlet(beta+zeros(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    
    #step 2. draw observations
    for n in range(N):
        E[n] = random.binomial(1,r)
        if E[n]==0:
            if EXTRA_VERBOSE: print 'e_0',E[n]
            prev_loc = X[n-1,:].argmax()
            pr_x = mu[prev_loc,:]
        else:
            if EXTRA_VERBOSE: print 'e_1',E[n]
            #uniform random choice:
            pr_x = ones(L)/float(L)
        X[n,:] = random.multinomial(1, pr_x)
        l_n = X[n,:].argmax()
        #finally, draw time context using location and departure random variable instantiations:
        if E[n]==0:
            #then day/time is drawn from parameters associated with departure from routine:
            #(N.B., w[1,i,:] is not used for i>0
            pr_h = w[0,0,:]
        else:
            #then day/time is drawn from parameters associated with current location (l_n):
            pr_h = w[1,l_n,:]
        D[n,:] = random.multinomial(1, pr_h)
    
    return X,D,E,w,r,mu

if __name__ == "__main__":
    X,D,E,w,r,mu = gen(200,8)
    lines = '\n---------------------------------------------------\n\n'
    print lines
    print 'PARAMETERS:'
    print 'w',w
    print 'mu',mu
    print 'r',r
    print lines
    print 'LATENT VARIABLES:'
    print 'E (departures)',E
    print lines
    print 'OBSERVATIONS:'
    print 'X (location observations)',X
    print 'D (time observations)',D
    print lines