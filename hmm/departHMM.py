'''
Created on 28 Mar 2013

@author: James McInerney
'''
from numpy import *
from em_tools import optT, optQ, gen, makeCoarse, calcME
from hmm.util import arrayWithOne, nearv, PARAMROOT, USERDATA_DIR, makeCoarse,\
    confusion
from matplotlib.pyplot import *
from hmm import em
from hmm.em import viterbi


#-------- FUNCTIONS DEFINING BEHAVIOUR OF OBSERVATIONS IN HMM ----------------

def initO(L,NHW,NS,seed):
    seed = random.choice(1000)[0]
    return {'mu':initMu(L,seed), 'omega':initOmega(L,NHW,NS,seed)}

def initMu(L,seed):
    #generate random transition matrix (i.e. pr(l_n | l_n-1, e_n))
    random.seed(seed)
    alpha = 0.5 #prior count, used only for initialization (not inference/learning) 
    Mu_prior = alpha*ones((L,L)) + eye(L)
    #no need to model e_n factor, since that is hard coded (i.e., done in likO and optO)
    mu_routine = array([random.dirichlet(Mu_prior[lprev,:]) for lprev in range(L)])
    return mu_routine

def initOmega(L,NHW,NS,seed):
    #generate random matrix describing temporal observations given departure state and (possibly) current location
    random.seed(seed)
    beta = 0.5 #prior count, used only for initialization (not inference/learning)
    omega = array([[random.dirichlet(beta*ones(NHW)) for _ in range(L)] for _ in range(NS)])
    return omega

def initZ(N,NS):
    return array([[random.multinomial(1,ones(NS)/float(NS))] for _ in range(N)])

def likO(e_n, (l_n,l_prev,d_n), O):
    #likelihood given state e_n (coice of [0,1]) and observations and params O
    mu_routine,omega = O['mu'],O['omega']
    L = len(mu_routine[0,:]) #number of locations
    if e_n==0:
        #then completely in routine:
        lik = mu_routine[l_prev,l_n] * omega[0,l_n,d_n]
    elif e_n==1:
        #out of routine:
        lik = (1/float(L)) * omega[1,0,d_n] 
    return lik

def optOFnc(T_prior,beta,L,NHW):
    return lambda xs,E: optO(xs,E,T_prior,beta,L,NHW)

def optO(xs,E,T_prior,beta,L,NHW):
    #format: lambda xs,Z0: optOMult(xs,Z0,NW)
    #1. convert xs (time/space observations) to X,D
    ls = zip(*xs)[0]
    ds = zip(*xs)[2]
    X = array([arrayWithOne(l,L) for l in ls])
    D = array([arrayWithOne(d,NHW) for d in ds])
    #since E is Nx2, we can take just the second column to provide the same info (and this is what the opt functions expect)
    return {'mu':optT(X, E[:,1], T_prior), 'omega':optQ(D, E[:,1], X, NHW, beta)}

#--------------------------------------------------------


def test(useSynth=True,uid_i=0,ws=20,d_thres=400,t_thres=360,maxIter=50,coarseFactor=1,doPlot=True,saveResults=True,paramname=None,numRestarts=3,hypers=None):
    NHW=168*(60/float(ws))
    #HMM parameters:
    NS = 2 #number of states
    alpha, beta = 0.5, 0.5 #prior pseudocounts
    iv = ones(2)/2. #iv: initial distribution (vector of size NS) over states
    initLoc = 0
    M_prior = 1.*ones((NS,NS)) + 1.*eye(NS) #prior pseudocounts for transition matrix M (for hidden states in HMM)
    #alternative prior: M_prior = 2.*ones((NS,NS)) + 0.*eye(NS) #prior pseudocounts for transition matrix M (for hidden states in HMM)

    if useSynth:
        print 'using synthetic data'
        X,D,H,E_grnd,S_grnd_mu,S_grnd_sigma,Q_grnd,M_grnd,T_grnd = genHMM(400,10,2,M_fix=array([[0.95, 0.05],[0.5, 0.5]]),DW=NHW)
        print 'M_grnd',M_grnd
    else:
        print 'testing uidi',uid_i
        filename = USERDATA_DIR() + 'history_uidi%i_nhw%i_ws%i_dthr%i_tthr%i' % (uid_i,NHW,ws,d_thres,t_thres)
        X,D = load(filename + '_X.npy'), load(filename + '_D.npy')
        print 'X',X[:3,:]
        print 'D',D[:3,:]
        print 'shape(X)',shape(X)
        print 'shape(D)',shape(D)
        #make D coarser by factor |coarseFactor|:
        #coarseFactor = 2
        D = makeCoarse(D,(60/ws)*coarseFactor) 
    (N,L) = shape(X)
    
    ##------- DEFINE HYPERPARAMETERS -----------
    if hypers is None:
        M_prior = 1.*ones((NS,NS)) + 1.*eye(NS) #prior pseudocounts for transition matrix M (for hidden states in HMM)
        T_prior = 0.1*ones((L,L)) + 1.*eye(L) #prior for location transitions 
    else:
        #linear weighted sums (for 'stickiness' of states):
        mweights, tweights, beta = hypers['M_prior'], hypers['T_prior'], hypers['beta']
        M_prior = mweights[0]*ones((NS,NS)) + mweights[1]*eye(NS)
        T_prior = tweights[0]*ones((L,L)) + tweights[1]*eye(L)
        
    ##------------------------------------------
    
    #convert X, D to single list
    xs = zip(X.argmax(axis=1), [initLoc] + list(X[:-1,:].argmax(axis=1)), D.argmax(axis=1))
    print 'xs',xs[:3]
    print 'X',X[:3,:]
    assert len(D[:,0])==N
    assert len(xs)==N,'%i neq %i'%(len(xs),N)
    #O: initial observation parameters
    #lik(z,x,O): likelihood function for observations X, given current O
    #optO: function that updates O given information in current iteration (probably using maximum likelihood) 70!
    E,Mbest,Obest = em.run(xs,NS,iv,lambda seed:initO(L,NHW,NS,seed),lambda z,x,O : log(likO(z,x,O)),optOFnc(T_prior,beta,L,NHW),M_prior=M_prior,Zinit=lambda : initZ(N,NS),maxItr=maxIter,numRestarts=numRestarts,plotConverg=0,showSteps=0,pauseBetweenRestarts=0) 
    #find most likely sequence of departure from routine using Viterbi algorithm:
    print 'Mbest',Mbest
    Ebest = viterbi(xs,Mbest,Obest,likO,iv)
    #ascertain error for the different solutions:
    if useSynth:
        print 'ME (most likely path)',calcME(E_grnd,Ebest)
        print 'ME (inferred)',calcME(E_grnd,E[:,1])
        
    #save the results
    if saveResults:
        if paramname is None: paramname = makeParamName(uid_i,NHW,ws,d_thres,t_thres)
        saveParams(paramname,E,Ebest,Mbest,Obest)
    
    
    if doPlot:
        if useSynth: print 'plotting synth',E_grnd; scatter(range(N),E_grnd,color='r',marker='o')
        print 'E',E
        assert N==len(E[:,1]), (N,len(E[:,1]))
        scatter(range(N),E[:,1],marker='x',color='b')
        scatter(range(N),Ebest,marker='|',color='g')
        gca().xaxis.set_ticks(arange(0,N,5))
        gca().grid(True)
        xlim(0,200)
        if useSynth: ylim(-0.05,1.05)
        else: ylim(E[:,1].min(),E[:,1].max())
        
        if useSynth:
            #decide the binary class based on simple threshold (can be extended to include utility of FP, TP etc.):
            E_class = E[:,1]>(0.8*E[:,1].mean()) #set fairly low threshold here for example
            print 'E_class',E_class
            #calculate key stats:
            print 'RESULTS against (synthetic) ground truth\n----------------'
            conf = confusion(E_grnd,E_class)
            print 'confusion matrix\n',conf
            print 'precision:',100*conf[1,1]/conf[:,1].sum(),'%'
            print 'recall:',100*conf[1,1]/conf[1,:].sum(),'%'
            
        show()



def makeParamName(uid_i,NHW,ws,dthr,tthr):
    s = 'uidi%i_nhw%i_ws%i_dthr%i_tthr%i' % (uid_i,NHW,ws,dthr,tthr)
    return s

def saveParams(paramname,E,Ebest,M,O):
    save(PARAMROOT() + 'E_' + paramname, E)
    save(PARAMROOT() + 'Ebest_' + paramname, Ebest)
    save(PARAMROOT() + 'M_' + paramname, M)
    save(PARAMROOT() + 'O_' + paramname, O)

def loadParams(paramname):
    E = load(PARAMROOT() + 'E_' + paramname + '.npy')
    Ebest = load(PARAMROOT() + 'Ebest_' + paramname + '.npy')
    M = load(PARAMROOT() + 'M_' + paramname + '.npy')
    O = load(PARAMROOT() + 'O_' + paramname + '.npy')
    return E,Ebest,M,O
    

def genHMM(N,L,NS,initE=0,M_prior=None,M_fix=None,DW=24,EXTRA_VERBOSE=False):
    #loc, day of week, hour of day, departure from routine:
    X,D,H,E = zeros((N,L)), zeros((N,DW)), zeros(N), zeros(N)
    #priors:
    alpha,beta,delta,gamma = 0.2,1,2,2
    mu_mu, mu_sigma = 12., 5.
    sigma_conc = 5.0
    #generative model:
    #step 1. draw parmeters from priors
    T = array([random.dirichlet(alpha+zeros(L)) for _ in range(L)]) #transition matrix p(l_n | l_n-1)
    Q = array([[random.dirichlet(beta+zeros(DW)) for _ in range(L)] for _ in range(2)]) #p(d_n | e_n, l_n)
    S_mu = array([[random.normal(mu_mu,mu_sigma) for _ in range(L)] for _ in range(2)])
    S_sigma = array([[random.gamma(sigma_conc) for _ in range(L)] for _ in range(2)])
    if M_prior is None: M_prior = ones((NS,NS)) #uniform prior pseudo counts for hidden state transitions
    if M_fix is None: M = array([random.dirichlet(M_prior[e_prev,:]) for e_prev in range(NS)])
    else: M = M_fix
    
    #step 2. draw observations
    for n in range(N):
        if n>0: e_prev = E[n-1]
        else: e_prev = initE
        E[n] = random.multinomial(1,M[e_prev,:]).argmax()
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
    
    return X,D,H,E,S_mu,S_sigma,Q,M,T


def keyStats(uid_i,ws=20,d_thres=400,t_thres=360,paramname=None):
    NHW=168*(60/float(ws))
    print 'key stats uidi',uid_i
    if paramname is None: paramname = makeParamName(uid_i,NHW,ws,d_thres,t_thres)
    E,Ebest,M,O = loadParams(paramname)
    filename = USERDATA_DIR() + 'history_uidi%i_nhw%i_ws%i_dthr%i_tthr%i' % (uid_i,NHW,ws,d_thres,t_thres)
    X,D = load(filename + '_X.npy'), load(filename + '_D.npy')
    #print 'X',X[:5,:]
    #print 'D',D[:5,:]
    print 'shape(X)',shape(X)
    print 'shape(D)',shape(D)
    print 'M',M
    print 'stationary distribution over states:'
    (evals,evecs) = linalg.eig(M.T)
    print 'evals',evals
    print 'evecs',evecs,shape(evecs)
    (i,) = where(nearv(evals,1.))
    print 'i',i
    v = evecs[:,i]
    #normalise:
    v= v/v.sum()
    print 'v',v,shape(v)
    v = reshape(v,(1,2))
    print 'v',v
    v1=dot(v,M)
    print '\tcheck',v1
    print '\n\n'
    save(PARAMROOT() + 'stdist_' + paramname, v.flatten())
    

if __name__ == "__main__":
    test(uid_i=0,ws=60,d_thres=500,useSynth=1,numRestarts=3,saveResults=0)
    
    