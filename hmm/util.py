'''
Created on 12 Mar 2013

@author: root
'''

from numpy import *

def rotateLeft(v,i):
    #v[j] is now assigned to v[j-i]
    roll(v,-i)

def arrayWithOne(i,N):
    x = zeros(N)
    x[i]=1.0
    return x

def standardize(v):
    #v: vector (one dim)
    #subtract min, put all v_i in range [0,1]:
    v1 = v-v.min()
    return v1/v1.max()

def standardizeSd(v):
    #v: vector (one dim)
    #make v standard deviation 1. with mean 0.
    v1 = v - v.mean()
    sd = sqrt(v1.var())
    v1 = v1/sd
    return v1


def nearto(x,y,eps=1e-50):
    return abs(x-y)<eps

nearv = vectorize(nearto)

def logNorm(lg_x):
    #normalises lg_x (vector of len K)
    (K,)=shape(lg_x)
    lg_x -= lg_x.max()
    #get log_sum
    ls = -inf
    for k in range(K):
        ls = logaddexp(ls, lg_x[k])
    #subtract log_sum from all
    lg_x -= ls
    #check that it is normalised:
    x = exp(lg_x)
    assert (1.-x.sum())<0.00000001,x.sum()
    return lg_x


def logOfNormal(x,mu,sd):
    #to avoid underflow when dealing with gaussians:
    A = -log(sd) - 0.5*log(2*pi)
    B = -((x-mu)**2)/(2*sd**2)
    return A+B

def USERDATA_DIR():
    setupDir = ''
    plat = detectPlatform()
    print 'platform detected:',plat
    if plat=='lab': 
        setupDir = '/media/8A8823BF8823A921/Dropbox/Nokia_processed_data/'
    elif plat=='iridis': 
        setupDir = '/home/jem1c10/journal/data/history/'
    elif plat=='laptop': 
        setupDir = '/Users/James/Dropbox/Nokia_processed_data/'
    else:
        raise Exception('define explicit dir')
    return setupDir

def makeCoarse(D,crs):
    (N,NHW)=shape(D)
    D_crs = zeros((N,1+NHW/crs))
    for n in range(N):
        d_n = int(D[n,:].argmax()/crs)
        #print 'd_n',d_n
        #print 'D_crs[n,:] (%i)'%len(D_crs[n,:]),D_crs[n,:]
        D_crs[n,d_n]=1.
    return D_crs

def PARAMROOT():
    plat = detectPlatform()
    if plat=='lab':
        return '/media/8A8823BF8823A921/SAMSUNG_BACKUP_MAR13/journal_data/' #+ 'NEW/' #<------- DON'T FORGET THIS!!!
    elif plat=='laptop':
        return '/Users/James/Dropbox/journal_params/'
    elif plat=='iridis':
        return '/home/jem1c10/journal/data/params/'
    raise Exception('define root for parameters')

def confusion(Y_grnd, Y):
    #compares binary classification Y_grnd and Y and returns confusion matrix:
    (N,)=shape(Y_grnd)
    (N1,)=shape(Y)
    assert N==N1
    conf = zeros((2,2))
    for n in range(N): conf[Y_grnd[n],Y[n]] += 1
    return conf


def detectPlatform():
    #returns platform indicator:
    try:
        open('/home/james/todo')
        #conclusion: lab computer
        return 'lab'
    except IOError:
        0
    try:
        open('/home/jem1c10/isIridis')
        #conclusion: iridis
        return 'iridis'
    except IOError:
        0
    try:
        open('/Users/James/laptop')
        #conclusion: laptop
        return 'laptop'
    except IOError:
        raise Exception('dont know root folder')