'''
Created on 12 Mar 2013

@author: James McInerney
'''

#general HMM implementation, with cusomtizable likelihood functions for observations

from numpy import *
from hmm.util import arrayWithOne, logNorm
from matplotlib.pyplot import *
from hmm.multinomial_obs import optOMult, margForOMult, logMargForOMult,\
    logOptOMult
from hmm.benchmarkos import FirstOrderMM, RandomBench
import time
from matplotlib.pyplot import *


seterr(all='ignore')




def predict(xs_test,NS,M,O,iv,lik,margForO):
    #returns predictive density over observations given recent history xs_test
    #M: learnt transition matrix for HMM
    #O: observation parameters
    #iv: initial dist. over states
    #lik: likelihood function
    #optO: 
    
    #(K,NW)=shape(O) #number of states, number of words
    
    #to predict, we need the distribution over the final state in the given test data:
    lnalphas = lnForwardsRecursion(xs_test,lik,M,NS,O,iv)
    #normalise final distribution:
    lg_z_n = lnalphas[-1,:]
    lg_z_n -= lg_z_n.max()
    z_n = exp(lg_z_n)/exp(lg_z_n).sum()
    #marginalise over prev to get distribution over next state:
    z_next = zeros(NS)
    for k in range(NS): z_next += z_n[k]*M[k,:]
    #finally, marginalise over next state to get observation distribution:
    pr_obs = margForO(z_next,O)
    return pr_obs

def logPredict(xs_test,NS,M,O,iv,lik,logMargForO):
    #returns log predictive density over observations given recent history xs_test
    #M: learnt transition matrix for HMM
    #O: observation parameters
    #iv: initial dist. over states
    #lik: likelihood function
    #optO: 
    
    #(K,NW)=shape(O) #number of states, number of words
    
    #to predict, we need the distribution over the final state in the given test data:
    lnalphas = lnForwardsRecursion(xs_test,lik,M,NS,O,iv)
    #normalise final distribution:
    lg_z_n = lnalphas[-1,:]
    log_zn = logNorm(lg_z_n)
    #marginalise over prev to get distribution over next state:
    z_next = zeros(NS)-inf
    for k in range(NS):
        for k_next in range(NS):
            z_next[k_next] = logaddexp(z_next[k_next], log_zn[k]+log(M[k,k_next]))
    #finally, marginalise over next state to get observation distribution:
    log_pr_obs = logMargForO(z_next,O)
    return log_pr_obs



    
def run(xs,NS,iv0,Oinit,loglik0,optO,Zinit=None,M_prior=None,M_fixed=None,Z_fixed=None,showSteps=True,plotConverg=False,maxItr=300,numRestarts=3,pauseBetweenRestarts=0):
    #xs: list of observations for each time step
    #NS: number of states (could be numeric number, or array, indicating that we want to capture several states)
    #iv: initial distribution (vector of size NS) over states
    #O: observation parameters
    #lik(z,x,O): likelihood function for observations X, given current O
    #optO: function that updates O given information in current iteration (probably using maximum likelihood)

    N = len(xs)

    try:
        NS1 = list(NS) 
        P = len(NS1) #multiple states we need to keep track of
        iv = iv0
        loglik = loglik0
        flatStates = 0
    except TypeError:
        P = 1 #for backwards compatability: able to provide scalar for NS (which is actually the most common use case)
        NS = array([NS])
        iv = zeros((P,NS))
        iv[0,:] = iv0
        loglik = lambda p,k,x,O: loglik0(k,x,O) #irrelevant what p is, because there is only one possible value
        flatStates = 1

    logLiks,bestRun = zeros(numRestarts)-inf, 0
    Zbest,Mbest,Obest = None, None, None
    O = Oinit(random.choice(1000)[0])
    for restart in range(numRestarts):
        if restart>0:
            print 'O_pre',O
            print 'M_pre',M
            print 'Z_pre',Z
            Z2=Z.copy()
            O2=O['omega'].copy()
            M2=M.copy()

        #initialize the model params:
        #transition matrix M:
        if M_fixed is None: M = array([random.dirichlet(ones(NS[0])) for _ in range(NS[0])])
        else: M = M_fixed.copy()
        seed = random.choice(1000)[0]
        print 'seed',seed
        O = Oinit(seed)
        #if Zinit is not None: Z = Zinit()
        Z = array([[random.multinomial(1,ones(NS[0])/float(NS[0]))] for _ in range(N)])
    

        #do until convergence
        itr,totalLogLik_old,diff=0,-9999999,9999999
        log_liks = []
        while itr<maxItr and abs(diff)>0.0001:
            print 'iteration',itr
            if restart>0: 
                assert not(all(Z==Z2))
                assert not(all(O['omega']==O2))
                #assert not(all(M==M2))

            #E-step:
            if True: #Zinit is None
                #1. forward recursion:
                lnalphas = lnForwardsRecursion(xs,loglik,M,NS,O,iv,P)
                #2. backward recursion:
                lnbetas = lnBackwardsRecursion(xs,loglik,M,NS,O,P)
                #3. calculate exp(Z):
                Z_,lnXlik = calcExpZ(lnalphas,lnbetas)
                if Z_fixed is None: Z = Z_
                else: Z = Z_fixed
                
                
                if showSteps:
                    print 'alphas_%i'%itr,lnalphas
                    print 'betas_%i'%itr,lnbetas            

            #M-step
            #prepare latent variables for M-step: remove one dimension if necessary (to make optimisation easier for user)
            if flatStates: Z1 = Z[:,0,:] #Z.f#Z1 = reshape(Z,(N,NS[0]))
            else: Z1 = Z

            #update transition matrix:
            if M_fixed is None: M = calcM2(Z,iv,lnalphas,lnbetas,M_prior=M_prior)
                
            #update observation params:
            NHW=168*(60/float(20))
            L=20
            beta=0.5
            T_prior = 0.1*ones((L,L)) + 1.*eye(L)
            O = optO(xs,Z1)

            
            if showSteps:
                print 'O_%i'%itr,O
                print 'M_%i'%itr,M
                print 'Z_%i'%itr,Z
            
            if Zinit is None or itr>0: 
                totalLogLik = lnXlik
                log_liks.append(totalLogLik)
                diff = totalLogLik-totalLogLik_old
                print 'total log likelihood',totalLogLik
                assert diff>=-2., 'diff %f: %f < %f' % (diff,totalLogLik, totalLogLik_old)
                totalLogLik_old = totalLogLik
            itr+=1
            
        if plotConverg or restart==0:
            plot(range(len(log_liks)), log_liks)
            xlabel('iterations')
            ylabel('total log likelihood of data')
            show()
        
        #if log lik better than best run, then use the results of this run as final result (for now)
        if totalLogLik>logLiks[bestRun]:
            bestRun = restart
            Zbest, Mbest, Obest = Z,M,O
        logLiks[restart] = totalLogLik
        time.sleep(pauseBetweenRestarts)
        print 'O_%i'%itr,O
        print 'M_%i'%itr,M
        print 'Z_%i'%itr,Z


    print 'AFTER %i RESTARTS, USING RESULTS FROM RUN %i'%(numRestarts,bestRun)


    if flatStates: return Zbest[:,0,:],Mbest,Obest
    else: return Zbest,Mbest,Obest
    

def forwardsRecursion(xs,lik,M,K,O,iv):
    N = len(xs)
    assert len(iv)==K
    (K1,K2) = shape(M)
    assert K1==K
    assert K2==K
    
    #these store the forward recursion probability distribution over states for each time step in N
    alphas = zeros( (N,K) )
    
    #start off the chain with initial condition
    for k in range(K): alphas[0,k] = iv[k]*lik(k,xs[0],O)
    #normalise:
    alphas[0,:] = alphas[0,:]/alphas[0,:].sum()
    #print 'lik xs[0]',[lik(k0,xs[0],O) for k0 in range(K)]
    
    
    for n in range(1,N):
        for k_next in range(K):
            #marginalise over previous states to get p(z_{n,k_next})
            prZn = 0.0
            for k_prev in range(K):
                kval = alphas[n-1,k_prev]*M[k_prev,k_next]
#                print 'k_prev,kval',k_prev,kval
                prZn += kval
            alphas[n,k_next] = lik(k_next,xs[n],O)*prZn
        #have to make sure alphas are normalised as we progress (really?)
        #alphas[n,:] = alphas[n,:]/alphas[n,:].sum()

    return alphas

def backwardsRecursion(xs,lik,M,K,O):
    N = len(xs)
    betas = zeros((N,K))
    
    #set initial condition (for state at time step n)
    betas[N-1,:] = ones(K)
    
    for n in arange(N-2,-1,-1): #from [N-2,N-3,...,0]
        for k in range(K):
            prXZn = 0.0
            #marginalise over next state z_n+1
            for k_next in range(K):
                #print '(backwards) lik(k_%i,xs[%i],O)'%(k_next,n+1),lik(k_next,xs[n+1],O)
                prXZn += betas[n+1,k_next]*lik(k_next,xs[n+1],O)*M[k,k_next]
            betas[n,k] = prXZn
        #normalise:
        #betas[n,:]=betas[n,:]/betas[n,:].sum()
    return betas


def lnForwardsRecursion(xs,loglik,M,NS,O,iv,P):
    N = len(xs)
    assert all(NS[0]==NS) #all same
    (P,K)=shape(iv)
    assert K==NS[0]
    (K1,K2) = shape(M)
    assert K1==K
    assert K2==K
    
    #these store the forward recursion probability distribution over states for each time step in N
    alphas = zeros( (N,P,K) )
    
    for n in range(N):
        for p in range(P):
            for k_next in range(K):
                #marginalise over previous states to get p(z_{n,k_next})
                lnPrZn = -inf
                for k_prev in range(K):
                    if n==0: prevAlphas = log(iv[p,k_prev])
                    else: prevAlphas = alphas[n-1,p,k_prev]
                    
                    kval = prevAlphas+log(M[k_prev,k_next])
    #                print 'k_prev,kval',k_prev,kval
                    lnPrZn = logaddexp(lnPrZn,kval)
                lnPrObs = loglik(p,k_next,xs[n],O)
                alphas[n,p,k_next] = lnPrObs+lnPrZn

    print 'alphas',alphas
    return alphas

def lnBackwardsRecursion(xs,loglik,M,Ks,O,P):
    N = len(xs)
    K = Ks[0]
    betas = zeros((N,P,K))
    
    #set initial condition (for state at time step n)
    betas[N-1,:,:] = zeros((P,K))
    
    for n in arange(N-2,-1,-1): #from [N-2,N-3,...,0]
        for p in range(P):
            for k in range(K):
                lnPrXZn = -inf
                #marginalise over next state z_n+1
                for k_next in range(K):
                    #print '(backwards) lik(k_%i,xs[%i],O)'%(k_next,n+1),lik(k_next,xs[n+1],O)
                    lnPrXZn = logaddexp(lnPrXZn, betas[n+1,p,k_next]+loglik(p,k_next,xs[n+1],O))+log(M[k,k_next])
                betas[n,p,k] = lnPrXZn
            #normalise:
            #betas[n,:]=betas[n,:]/betas[n,:].sum()
    return betas


def calcM1(Z,iv,M_prior=None):
    #M_prior: prior pseudocounts for transition matrix M
    (N,P,K)=shape(Z)
    if M_prior is None: 
        print 'using ML for M'
        E = zeros((K,K))
    else: E = M_prior
    for n in range(N):
        for p in range(P):
            for k_prev in range(K):
                if n==0: prevZ = iv[p,k_prev]
                else: prevZ = Z[n-1,p,k_prev]
                v = prevZ*Z[n,p,:]
                E[k_prev,:] += v
    #normalise E to get M
    M = E/reshape(E.sum(axis=1), (K,1))
    return M


def calcM2(Z,iv,lnalphas,lnbetas,M_prior=None):
    #M_prior: prior pseudocounts for transition matrix M
    (N,P,K)=shape(Z)
    if M_prior is None: 
        print 'using ML for M'
        E = zeros((K,K))
    else: E = M_prior
    for n in range(1,N):
        for p in range(P):
            for k_prev in range(K):
                v = Z[n-1,p,k_prev]*Z[n,p,:]
                E[k_prev,:] += v
    #normalise E to get M
    M = E/reshape(E.sum(axis=1), (K,1))
    return M

def calcEVals(alphas,betas,xs,O,M,Xlik,lik):
    N1 = len(xs)
    (N,K) = shape(alphas)
    (N2,K2) = shape(betas)
    (K3,K4)=shape(M)
    assert N==N1
    assert N==N2
    assert K==K2
    #assert N==len(Xlik),'N=%i,len(Xlik)=%i'%(N,len(Xlik))
    assert K3==K4
    assert K3==K
    
    evals = zeros((N-1,K,K))
    for n in range(1,N):
        for k_prev in range(K):
            for k in range(K):
                try:
                    evals[n-1,k_prev,k] = (alphas[n-1,k]*lik(k, xs[n], O)*M[k_prev,k]*betas[n,k])/Xlik
                except ValueError:
                    print 'alphas',alphas[n-1,k]
                    print 'obsLik',lik(k, xs[n], O)
                    print 'R',M[k_prev,k]
                    print 'betas',betas[n,k]
                    print 'Xlik',Xlik
                    raise Exception()

    return evals

def calcM(E):
    (N,K,K1)=shape(E)
    assert K1==K

    #ML estimate of transition matrix
    F1 = E.sum(axis=0)
    #normalise
    F = F1/reshape( F1.sum(axis=1), (K,1) )
    return F

def calcExpZ(lnalphas,lnbetas):
    (N,P,K) = shape(lnalphas)
    (N1,P1,K1) = shape(lnbetas)
    assert N==N1
    assert K1==K
    assert P==P1 #number of states we need to keep track of (different from K, which is the total number of possible states)
       
    Z = zeros( (N,P,K) )
    
    for n in range(N):
        for p in range(P):
            lnz = lnalphas[n,p,:]+lnbetas[n,p,:]
            #normalise:
            lnz -= lnz.max()
            z = exp(lnz)
            #print 'z,sum',z,z.sum()
            Z[n,p,:] = z/z.sum()
        
    Xlik = -inf
    for k in range(K):
        for p in range(P):
            Xlik = logaddexp(Xlik, lnalphas[-1,p,k])
    
    return Z,Xlik
        

def near(x,y,thresh=0.0000000001):
    diff = abs(x-y)
    return diff<thresh

    
def gen(N,NS,NW,show=True,fullObs=False):
    #generate N discrete observations:
    
    #generate transition matrix
    M = array([random.dirichlet(0.1*ones(NS)) for _ in range(NS)]) #array([[0.,1.],[1.,0.]]) #ones((NS,NS))/float(NS) # 
    #generate observation matrix, given word size NW
    X = zeros((N,NW))
    if fullObs:
        assert NW==NS,'states must eq obs for full obs'
        O = eye(NW)
    else:
        O = array([random.dirichlet(0.5*ones(NW)) for _ in range(NS)]) #pr(x | z) #eye(NW)
    #generate Z matrix
    Z = zeros((N,NS))
    iv = ones(NS)/float(NS)
    #generate initial state and obs
    Z[0,:] = random.multinomial(1,iv)
    X[0,:] = random.multinomial(1,O[Z[0,:].argmax(),:])
    
    for n in range(1,N):
        #step 1. generate latent state, given previous state
        z_prev = Z[n-1,:].argmax()
        Z[n,:] = random.multinomial(1,M[z_prev,:])
        #step 2. given the current state, generate observation
        z = Z[n,:].argmax()
        X[n,:] = random.multinomial(1,O[z,:])
    
    if show:
        print 'O ground',O
        print 'M ground',M
        
    return X,Z,M,O


def viterbi(xs,M,O,lik,iv,VERBOSE=False):
    #returns the most likely sequence of hidden states (given the observations) using the Viterbi algorithm
    if VERBOSE: print 'iv',iv
    N = len(xs)
    (NS,NS1)=shape(M)
    assert NS==NS1
    #initialize T1 and T2
    T1,T2 = zeros((N,NS)), zeros((N,NS))
    for s in range(NS):
        T1[0,s] = iv[s]*lik(s,xs[0],O)
    
    #go through data and update T1 and T2
    for n in range(1,N):
        for s in range(NS):
            #v = p(z_n-1 = k) * p(z_n = s | z_n-1 = k) * p(x_n | z_n = s)
            #  = p(x_n, z_n = s, z_n-1 = k)
            #  (where s and x_n are given)
            v = (T1[n-1,:] * M[:,s] * lik(s,xs[n],O))
            if VERBOSE: print 'v_ns,lik',v,lik(s,xs[n],O)
            T1[n,s] = v.max() #find max_k p(x_n ,z_n=s, z_n-1=k)
            T2[n,s] = v.argmax() #find argmax_k p(x_n, z_n=s, z_n-1=k)
        #normalise:
        T1[n,:] = T1[n,:]/T1[n,:].sum()
    if VERBOSE:
        print 'T2',T2
        print 'T1',T1
    #find the most likely final state:
    S = zeros(N)
    S[N-1] = T1[N-1,:].argmax()
    for n in range(N-1,1,-1):
        S[n-1] = T2[n,S[n]]
    return S
            

def viterbiLog(xs,P,M,O,loglik,iv,VERBOSE=False):
    #returns the most likely sequence of hidden states (given the observations) using the Viterbi algorithm
    if VERBOSE: print 'iv',iv
    N = len(xs)
    (NS,NS1)=shape(M)
    assert NS==NS1
    #initialize T1 and T2
    T1,T2 = zeros((N,P,NS)), zeros((N,P,NS))
    for p in range(P):
        for s in range(NS):
            T1[0,p,s] = log(iv[p,s])+loglik(p,s,xs[0],O)
    
    #go through data and update T1 and T2
    for n in range(1,N):
        for p in range(P):
            for s in range(NS):
                #v = p(z_n-1 = k) * p(z_n = s | z_n-1 = k) * p(x_n | z_n = s)
                #  = p(x_n, z_n = s, z_n-1 = k)
                #  (where s and x_n are given)
                v = (T1[n-1,p,:] + log(M[:,s]) + loglik(p,s,xs[n],O))
                #if VERBOSE: print 'v_ns,lik',v,lik(s,xs[n],O)
                T1[n,p,s] = v.max() #find max_k p(x_n ,z_n=s, z_n-1=k)
                T2[n,p,s] = v.argmax() #find argmax_k p(x_n, z_n=s, z_n-1=k)
            #normalise:
            T1[n,p,:] -= T1[n,p,:].max()
            print 'T1',T1[n,p,:]
            T1[n,p,:] = log(exp(T1[n,p,:])/exp(T1[n,p,:]).sum())
    if VERBOSE:
        print 'T2',T2
        print 'T1',T1
    #find the most likely final state:
    S = zeros((N,P))
    for p in range(P):
        S[N-1,p] = T1[N-1,p,:].argmax()
        for n in range(N-1,1,-1):
            S[n-1,p] = T2[n,p,S[n,p]]
    return S
            
            
def test():
    NS,NW = 5, 6
    X,Z,M_ground,O_ground = gen(100,NS,NW)
    print 'generated X',X.argmax(axis=1)
    print 'generated Z (ground)',Z.argmax(axis=1)
    #prepare observation specific params for HMM:
    iv = ones(NS)/float(NS)
    lik = lambda z,x,O: O[z,x]
    #random initializaton of O (pr(x|z))
    O_init = array([random.dirichlet(ones(NW)) for _ in range(NS)]) #pr(x | z)
    #O_init = ones((NS,NW))/float(NW) #array([[0.51,0.49],[0.49,0.51]]) #
    #O_init = eye(NW)
    
    Z_inf,M_inf,O_inf = run(X.argmax(axis=1),NS,iv,O_init,lik,lambda xs,Z0: optOMult(xs,Z0,NW))
    print 'Z inferred',Z_inf
    print 'O inferred',O_inf
    print 'M inferred',M_inf
    print '--------'
    print 'Z ground',Z.argmax(axis=1)
    print 'O ground',O_ground
    print 'M ground',M_ground

    pr_next_obs = predict(X[-20:,:].argmax(axis=1), NS, M_inf, O_inf, iv, lik, margForOMult)
    print 'pr_next_obs',pr_next_obs
    
def testPred():
    #compare against simple markov model on observations
    NS,NW = 10, 10
    N = 1000 #number of data points
    X_all,Z_all,M_ground,O_ground = gen(N,NS,NW)
    #split into training and testing data:
    trainSize = 0.2*N
    X, Z = X_all[:trainSize,:], Z_all[:trainSize,:]
    X_test, Z_test = X_all[trainSize:,:], Z_all[trainSize:,:]
    
    #train bench on same training data
    bench = FirstOrderMM()
    bench.train(X)
    benchRand = RandomBench()
    benchRand.train(X)
    
    print 'generated X',X.argmax(axis=1)
    print 'generated Z (ground)',Z.argmax(axis=1)
    #prepare observation specific params for HMM:
    iv = ones(NS)/float(NS)
    lik = lambda z,x,O: O[z,x] #takes account of fact that we keep the O params in log space
    #random initializaton of O (pr(x|z))
    O_init = log(array([random.dirichlet(0.5*ones(NW)) for _ in range(NS)])) #pr(x | z)
    #O_init = ones((NS,NW))/float(NW) #array([[0.51,0.49],[0.49,0.51]]) #
    #O_init = eye(NW)
    
    #
    #logOptOMult(xs,Z0,NW)
    #M_fixed=M_ground,  
    #log(O_ground)
    #M_fixed=M_ground,
    #logOptOMult(xs,Z0,NW),
    Z_ground = Z_all[:200,:]
    Z_ground = array(map(lambda v: [v], Z_ground)) 
    print 'shape Z_all',shape(Z_ground)
    print 'shape X',shape(X)
    #M_fixed=M_ground,
    #Z_fixed=Z_ground,  
    Z_inf,M_inf,O_inf = run(X.argmax(axis=1), NS, iv, lambda : O_init, lik, lambda xs,Z0: logOptOMult(xs,Z0,NW), maxItr=50, numRestarts=3,plotConverg=False,pauseBetweenRestarts=2)
    print 'Z inferred',Z_inf
    print 'O inferred',O_inf
    print 'M inferred',M_inf
    print '--------'
    print 'Z ground',Z.argmax(axis=1)
    print 'O ground',O_ground
    print 'M ground',M_ground

    testBlockSize = 10
    NB = int((N-trainSize-1)/testBlockSize)
    totalLogLik, totalLogLikBench, totalLogLikRand, totalLogLikUB = zeros(NB), zeros(NB), zeros(NB), zeros(NB)
    for bl in range(NB):
        sp,sp1 = int(bl*testBlockSize + trainSize), int((bl+1)*testBlockSize + trainSize)
        xs_test = X_all[sp:sp1,:].argmax(axis=1)
        print 'xs_test',xs_test
        print 'sp,sp1',sp,sp1
        log_pr_next_obs = logPredict(xs_test, NS, M_inf, O_inf, iv, lik, logMargForOMult)
        #compare against actual next obs
        x_ground = X_all[sp1,:].argmax()
        testLogLik = log_pr_next_obs[x_ground] #careful about last obs
        totalLogLik[bl] = testLogLik
        #now compare against benchmark
        log_pr_obs_bench = bench.logPredict(xs_test)
        totalLogLikBench[bl] = log_pr_obs_bench[x_ground]
        totalLogLikRand[bl] = benchRand.logPredict(xs_test)[x_ground]
        #finally, calc upper bound by using the ground truth latent state and taking log of obs matrix
        z_ground = Z_all[sp1,:].argmax()
        totalLogLikUB[bl] = log(O_ground[z_ground,:])[x_ground]
        
    print 'totalLogLik',totalLogLik.sum(),totalLogLik
    print 'totalLogLikBench',totalLogLikBench.sum(),totalLogLikBench
    print 'totalLogLikRand',totalLogLikRand.sum(),totalLogLikRand
    print 'totalLogLikUB (knowing the current state, but with noisy observation)',totalLogLikUB.sum(),totalLogLikUB
    
    
if __name__ == "__main__":
    testPred()