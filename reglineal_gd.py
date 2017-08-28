#!/usr/bin/python

from numpy           import array, arange, exp, log10, dot, vstack, hstack, zeros, ones
from numpy.random    import normal, randn, shuffle

def genera_ds(N=100, m=1, b=0, s=1):
    X = arange(N)
    Y = [m*x+b for x in X]
    Y += s*randn(N)
    return array( [ (x,y) for x,y in zip(X,Y) ]  )

def h(W,X):
    Yt = W.dot(X.T)
    return Yt

def J(W, X, h, Y):
    Yt = h(W,X)
    s = (0.5)*sum( (Yt-Y)**2 )
    return s

def gradiente(h,W,X,Y):
    n,m = X.shape
    grad = zeros((m,))
    for i in range(n):
        for j in range(m):
            grad[j] += (h(W,X) - Y)*X[i,j]
    return grad

def descenso_grad(w0in, J, h, Xorig, Y, epocas=1000, alfa=0.1, agregabias=True, verbose=True):
    if(agregabias):
        X = array([ hstack( (1,x) ) for x in Xorig ]) #agregamos un 1 al inicio del DS
    else:
        X = Xorig.copy()
    w0 = array(w0in)
    if(verbose):
        print("X es {0}".format( ','.join( map( str, X[:int(0.1*len(X))] ) ) ))
        print("Y es {0}".format( Y[:int(0.1*len(Y))] ))
        print("w0 es {0}".format( w0 ))
    W = [w0]
    E = [ J(w0, X, h, Y) ]
    m = len(w0)
    n = len(X)
    for epoca in range(epocas):
        wa = W[-1]
        wn = zeros(wa.shape)
        #error = Y-h(wa,X)
        error = 0.
        for i in range(n):
            ejemplo = X[i]
            yh = h(wn, ejemplo)
            err = yh - Y[i]
            error += err**2
            for j in range(m):
                wn[j] = wa[j] - alfa*error*ejemplo[j]
        #grad = dot(error,X)
        #wn = wa - alfa* grad
        W.append(wn)
        c = J(wn, X, h, Y)
        E.append( c )
        if(verbose):
            if(epoca%10==0): print("epoca {0} error {1} peso {2}".format(str(epoca),str(c),str(wn)))
    return W, E


