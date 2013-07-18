'''
Created on 6 Apr 2013

@author: root
'''


'''
Created on 18 Dec 2012

@author: James McInerney
'''

#import matplotlib
#matplotlib.use('Agg') #backend engine


#plots average roc curve for set of roc curves using linear interpolation
from matplotlib.pyplot import * #plot, show, figure, xlim, ylim, title, scatter
from numpy import * #array,random,arange,zeros
from random import choice
import itertools as itt

class Interpolation(object):
    def __init__(self,xs,ys):
        assert len(xs)==len(ys)
# assert self._monotonically_increasing(xs)
# assert self._monotonically_increasing(ys)
        xs1, ys1 = self._sort_xys(xs,ys) #throws error if xs,ys not monotonically increasing
        self.xs, self.ys = self._remove_dups(xs1,ys1) #remove duplicate x values (taking max of corresponding y values)
        self.min_x = self.xs[0]
        self.max_x = self.xs[-1] #get last element in list
    
    def _monotonically_increasing(self,vs):
        #returns True iff vs list is monotonically increasing (not necessarily strictly increasing)
        i = 0
        print 'vs',vs
        while vs[i]<=vs[i+1] and i<(len(vs)-1): i+=1
        return i==(len(vs)-1)
    
    def _sort_xys(self,xs,ys):
        xs_is = zip(xs, range(len(xs))) #zip indices with values
        xs1,is1 = zip(*sorted(xs_is))
        ys1 = []
        for i in is1: ys1.append(ys[i])
        return xs1,ys1

    def _remove_dups(self,xs,ys):
        #removes duplicate x values (taking max of corresponding y values)
        i = 0
        xs1,ys1 = [], []
        while i < len(xs):
            #search forwards for matching x values (could be more than 2)
            j = 1
            while (i+j) < len(xs) and xs[i]==xs[i+j]: j+=1
            y_max = max(ys[i:(i+j)])
            xs1.append(xs[i])
            ys1.append(y_max)
            i+=j
        return xs1,ys1
    
    def getY(self,x):
        #get y by direct value or linear interpolation
        assert x >= self.min_x and x <= self.max_x
        #find nearest x value
        i = 0
        while self.xs[i]<x: i+=1 #by previous assertion, we know that in worst case self.xs[N-1]==x
        if self.xs[i]==x:
            #just return direct value:
            return self.ys[i]
        else:
            #return linear interpolation:
            m = (self.ys[i]-self.ys[i-1])/(self.xs[i]-self.xs[i-1])
            return m*(x - self.xs[i-1]) + self.ys[i-1]
    
    def getAllY(self,xs):
        ys = []
        for x in xs:
            ys.append(self.getY(x))
        return ys

class RocSet(object):
    def __init__(self,xss,yss,weights=None):
        assert len(xss)==len(yss)
        self.rocs = []
        self.weights = weights
        for xs,ys in zip(xss,yss):
            self.rocs.append(Interpolation(xs,ys))
    
    def plot_values(self,N=100):
        interval = 1/float(N)
        xs = arange(0.,1.,interval)
        A = len(self.rocs) #number of approaches
        ys = zeros((A,N))
        assert len(xs)==N
        for roc,a in zip(self.rocs,itt.count()):
            ys[a,:] = array(roc.getAllY(xs))
        #ys = ys/float(len(self.rocs)) #average
        if self.weights is None:
            mn = ys.mean(axis=0)
            sd = sqrt(ys.var(axis=0))
            sem = sd/sqrt(A)
            eb = 1.96*sem
        else:
            print 'len self.rocs,weights',len(self.rocs),len(self.weights)
            mn = average(ys,axis=0,weights=self.weights)
            vr = dot(self.weights, (ys-reshape(mn,(1,N)))**2)/self.weights.sum()
            sem = sqrt(vr/float(A))
            eb = 1.96*sem
            print 'eb',shape(eb),eb
        return xs,mn,eb
    
def plotAve(xsss,ysss,lineNames,weights=None,selectors=None,titleStr='Average ROC',saveFile='',showIndividual=0,showScatter=0,doShow=1):
    
    #plots single roc corresponding to average of unsorted set of xs,ys (FP,TP)
    styles = ['-','--','-.',':','-']
    markers = ['x',None,None,None,None]
    for xss,yss,lineName,sty,marker,m in zip(xsss,ysss,lineNames,styles,markers,itt.count()):
        if weights is not None and selectors is not None:
            wgts = weights[where(selectors[m,:]==1.)]
        else: wgts = None
        roc_ave = RocSet(xss,yss,weights=wgts)
        xs,ys,eb = roc_ave.plot_values(N=50) #was 150
        errorbar(xs,ys,yerr=eb,label=lineName,linestyle=sty,marker=marker,markersize=4)
        if showScatter:
            for xs,ys in zip(xss,yss):
                #title('ROC %i'%i)
                scatter(xs,ys)
        i=1
        if showIndividual:
            for xs,ys in zip(xss,yss):
                figure()
                title('ROC %i'%i)
                scatter(xs,ys)
                xlim((0.,1.))
                ylim((0.,1.))
                i+=1
    gca().grid(which='both',color='gray') #,linestyle='--')
    xlim((0.,1.))
    ylim((0.,1.))
    xlabel('False positive rate')
    ylabel('True positive rate')
    title(titleStr)
    legend(loc='lower right')
    if saveFile is not None:
        savefig(saveFile,dpi=80,format='eps')
    if doShow: show()
    else: figure()
    
def genRandom(size=3):
    #generate |size| random roc curves (xs=FP, ys=TP)
    xss,yss = [], []
    for _ in range(size):
        xs = [0.0,1.0]
        xprev = 0.0
        for _ in range(10):
            xs.append(xprev + choice([0.0,random.beta(1,10)*(1.-xprev)])) #put in some duplicate xs, as may happen from time to time
            xprev = xs[-1]
        xss.append(xs)
    for _ in range(size):
        ys = [0.0,1.0]
        yprev = 0.0
        for _ in range(10):
            ys.append(yprev + random.beta(1,5)*(1.-yprev))
            yprev = ys[-1]
        yss.append(ys)
    return xss,yss

def test():
    xss,yss = genRandom()
    print 'xss',xss[:10]
    print 'yss',yss[:10]
    plotAve(xss,yss)

if __name__ == "__main__":
    test()