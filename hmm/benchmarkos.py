'''
Created on 13 Mar 2013

@author: root
'''

from numpy import *

class PredictBenchmark(object):
    def train(self,X):
        raise Exception('not implemented')
    
    def logPredict(self,xs_test):
        raise Exception('not implemented')
    
    
class FirstOrderMM(PredictBenchmark):
    
    def train(self,X):
        alpha = 0.5 #MAP approach
        xs = X.argmax(axis=1)
        (N,NW) = shape(X)
        #build up frequency transition matrix
        M = zeros((NW,NW)) + alpha
        x_prev = xs[0]
        for x in xs[1:]: 
            M[x_prev,x] += 1
            x_prev = x
        #normalise:
        M = M/reshape(M.sum(axis=1),(NW,1))
        self.M = M
            
    def logPredict(self,xs_test):
        #first order, so only interested in final obs
        x_prev = xs_test[-1]
        return log(self.M[x_prev,:])

class RandomBench(PredictBenchmark):
    
    def train(self,X):
        (N,self.NW) = shape(X)
        
    def logPredict(self,xs_test):
        lg_pr = zeros(self.NW)-log(self.NW)
        return lg_pr