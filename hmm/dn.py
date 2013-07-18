'''
Created on 11 Nov 2012
@author: James McInerney

Modified on 18 Apr 2013
@author: David Nicholson
'''



from numpy import *
from random import choice
from matplotlib.pyplot import *

seterr(all='raise')

def gen(N,L,DW=168,EXTRA_VERBOSE=False):
    #N: number of observations
    #L: number of discrete locations
    #DW: number of time slots per period (e.g., 1 week = 168 hours)
    
    #observations/latent variables: loc, day of week, departure from routine:
    X,D,E = zeros((N,L)), zeros((N,DW)), zeros((N,2)) # DaveN
    #priors:
    alpha,beta,delta,gamma = 1.0,1.0,1.0,1.0 
    #generative model:
    #step 1. draw parameters from priors
    mu = array([random.dirichlet(alpha+zeros(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    
    #r_in = array([random.beta(delta,gamma) for _ in range(2)]) # DaveN
    #r = vstack((r_in, 1-r_in)).T # DaveN #JM: added r_in to second array
    #JM: previous two lines commented out because it can be done more easily
    #and with more clarity about the roles of hyperparameters delta (for E[n]==1) and gamma (for E[n]==0)
    r = array([random.dirichlet([gamma,delta]) for _ in range(2)])
    
    w = array([[random.dirichlet(beta+zeros(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    
    #step 2. draw observations
    prev_E = choice([0,1]) #JM: keeps track of previous state of E. Initialise for n=-1 with uniform random choice.
    for n in range(N):
        #prev_E = E[N-1,:].argmax()          #DaveN #JM: approach is right, but should consider when n = 0 
        pr_e = r[prev_E,:]                  #DaveN
        E[n,:] = random.multinomial(1,pr_e) #DaveN
        
        if E[n,:].argmax()==0:              #DaveN
            if EXTRA_VERBOSE: print 'e_0',E[n,:] #Dave N
            prev_loc = X[n-1,:].argmax()
            pr_x = mu[prev_loc,:]
        else:
            if EXTRA_VERBOSE: print 'e_1',E[n,:] #DaveN
            #uniform random choice:
            pr_x = ones(L)/float(L)
        X[n,:] = random.multinomial(1, pr_x)
        l_n = X[n,:].argmax()
        #finally, draw time context using location and departure random variable instantiations:
        if E[n,:].argmax()==0: #JM: fixed
            #then day/time is drawn from parameters associated with current location (l_n):
            pr_h = w[0,l_n,:] #JM: fixed mistake in original version: E[n]==0 is in routine, E[n]==1 is departure
        else:
            #then day/time is drawn from parameters associated with departure from routine:
            #(N.B., w[1,i,:] is not used for i>0
            pr_h = w[1,0,:] #JM: fixed mistake in original version
        D[n,:] = random.multinomial(1, pr_h)
        prev_E = E[n,:].argmax()
    
    return X,D,E,w,r,mu

def genTime(N,L,DW=168,EXTRA_VERBOSE=False):
    #N: number of observations
    #L: number of discrete locations
    #DW: number of time slots per period (e.g., 1 week = 168 hours)
    
    #observations/latent variables: loc, day of week, departure from routine:
    X,D,E = zeros((N,L)), zeros((N,DW)), zeros((N,2)) # DaveN
    #priors:
    alpha,beta,delta,gamma = 1.0,1.0,1.0,1.0 
    #generative model:
    #step 1. draw parameters from priors
    mu = array([random.dirichlet(alpha+zeros(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    r = array([random.dirichlet([gamma,delta]) for _ in range(2)])    
    w = array([[random.dirichlet(beta+zeros(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    
    #step 2. draw observations
    prev_E = choice([0,1]) #JM: keeps track of previous state of E. Initialise for n=-1 with uniform random choice.
    for n in range(N):
        #next time step is deterministic:
        d_n = n % DW
        D[n,:] = arrayWithOne(d_n,DW)
        prev_loc = X[n-1,:].argmax()
        
        #p(e_n = 0 | e_prev, d_n) \propto sum_l ( p(e_n = 0 | e_prev) p(d_n | e_n = 0, l_n) p(l_n | e_n = 0, l_prev) )
        pr_e0 = sum([r[prev_E,0] * w[0,l,d_n] * mu[prev_loc,l] for l in range(L)])
        pr_e1 = r[prev_E,1] * w[1,0,d_n]
        pr_e = array([pr_e0, pr_e1]) #only two choices
        #normalise:
        pr_e = pr_e / pr_e.sum() 
        E[n,:] = random.multinomial(1,pr_e) #DaveN
        
        if E[n,:].argmax()==0:              #DaveN
            if EXTRA_VERBOSE: print 'e_0',E[n,:] #Dave N
            #p(l_n | l_prev, e_n=0, d_n) \propto p(l_n | l_prev) p(d_n | l_n, e_n = 0)
            pr_x = mu[prev_loc,:] * w[0,:,d_n]
            #normalise:
            pr_x = pr_x / pr_x.sum()
        else:
            if EXTRA_VERBOSE: print 'e_1',E[n,:] #DaveN
            #uniform random choice:
            pr_x = ones(L)/float(L)
        X[n,:] = random.multinomial(1, pr_x)
        prev_E = E[n,:].argmax()
    
    return X,D,E,w,r,mu

    
def arrayWithOne(i,N):
    x = zeros(N)
    x[i]=1.0
    return x

if __name__ == "__main__":
    X,D,E,w,r,mu = genTime(20,8)
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
    
    #plot results:
    (N,DW) = shape(D)
    scatter(range(N), X.argmax(axis=1))
    xlabel('time step')
    ylabel('location')
    show()
