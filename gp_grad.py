import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import gp3

def grad_matern_52(x,x_prime,l):
    # Assumes x and x_prime are size m x n for m points in n-d problem

    if x.ndim==1:
        x=x.reshape((1,len(x))) 
    if x_prime.ndim==1:
        x_prime=x_prime.reshape((1,len(x_prime)))
    r=np.linalg.norm(x-x_prime,axis=1)
    # print(r)
    c = -( 5/(3*l**2) + 5*np.sqrt(5)*r/(3*l**3))*np.exp(-np.sqrt(5)*r/l)
    # print(c)
    c=np.tile(c.reshape((len(c),1)),2)
    # print(c)
    gm52 = c * (x-x_prime) 
    return gm52

def grad_mu(x,train_x,train_f,k,grad_k):
    if x.ndim==1:
        x=x.reshape((1,len(x))) 
    if train_x.ndim==1:
        train_x=train_x.reshape((1,len(train_x)))
    if train_f.ndim==1:
        train_f=train_f.reshape((len(train_f),1))
    
    N=x.shape[0]
    M=train_x.shape[0]

    Kxx=gp3.cov_from_kernel(train_x, k)
    # print(Kxx)
    c=np.linalg.solve(Kxx, train_f);
    # print(c)

    dmu=np.zeros(x.shape)

    # Loop over x-values 
    for i in range(N):
        xi=np.tile(x[i,:].reshape((1,len(x[i,:]))),(M,1))
        # print("xi shape ",xi.shape)
        # print("train_x shape ",train_x.shape)
        dk=grad_k(xi,train_x)
        # print("dk shape ",dk.shape)
        # print(dk)
        dmu[i,:]=np.sum( dk * np.tile(c,2),axis=0)
        # print(dmu)

    return dmu

def grad_sigma(x,train_x,k,grad_k):
    if x.ndim==1:
        x=x.reshape((1,len(x))) 
    if train_x.ndim==1:
        train_x=train_x.reshape((1,len(train_x)))

    N=x.shape[0]
    M=train_x.shape[0]
    Kxx=gp3.cov_from_kernel(train_x, k)

    dsigma=np.zeros(x.shape)

    for i in range(N):
        xi=np.tile(x[i,:].reshape((1,len(x[i,:]))),(M,1))

        #compute vector of kernel evaluations: k(x*, x[i])
        ki=np.zeros((M,1))
        for j in range(M):
            ki[j]=k(train_x[j,:],x[i,:])
        # print('ki shape:', ki.shape)
        # print('Kxx shape:', Kxx.shape)
        
        c=np.linalg.solve(Kxx,ki)
        # print('c shape:',c.shape)

        sigma=np.sqrt(1-ki.T@c)
        # print('sigma:', sigma)

        dk=grad_k(xi,train_x)
        
        # print('dk shape:',dk.shape)
        dsigma[i,:] = -np.sum( dk * np.tile(c,2),axis=0)/sigma
    
    return dsigma

def grad_expected_improvement(x,train_x,train_f,k,grad_k):
    if x.ndim==1:
        x=x.reshape((1,len(x))) 
    if train_x.ndim==1:
        train_x=train_x.reshape((1,len(train_x)))
    if train_f.ndim==1:
        train_f=train_f.reshape((len(train_f),1))

    # Calculate the current best value f(x+) observed so far
    f_best = np.max(train_f)

    M=x.shape[0]
    grad_ei=np.zeros(x.shape)
    for i in range(M):
        xi=x[i,:].reshape((1,len(x[i,:])))
        [mu,sigma]=gp3.get_mean_and_sigma(xi,train_x,train_f,k)
        mu=mu.flatten()[0]
        sigma=sigma.flatten()[0]
        with np.errstate(divide='ignore'):
            imp = mu - f_best
            Z = imp / sigma
            grad_ei_i = norm.cdf(Z)*grad_mu(xi,train_x,train_f,k,grad_k) +  norm.pdf(Z) *grad_sigma(xi,train_x,k,grad_k)
            grad_ei_i[sigma == 0.0] = 0.0
        grad_ei[i,:]=grad_ei_i.flatten()

    return grad_ei

def nesterovAscent(x0,gradf,t,f):
    max_iters=10**3
    tol=1e-8
    err=[]
    x=copy.copy(x0)
    if x.ndim==1:
        x=x.reshape((1,len(x))) 
    y=copy.copy(x0)
    f_arr=[f(x)]
    for i in range(max_iters):
        residual=t*gradf(y)
        err.append(np.linalg.norm(residual))

        x_new=y+residual
        y=x_new+(i+1)/(i+4)*(x_new-x)
        x=copy.copy(x_new)
        
        f_arr.append(f(x))
        if err[i]<tol:
            break
        if i==max_iters-1:
            print("Max iterations reached in gradient ascent.")
    return x, err, f_arr

def test_2d_withGD():

    x = 8 * rng.random(20) - 4
    y = 8 * rng.random(20) - 4

    r1z = gp3.r1(x, y)
    r2z = gp3.r2(x, y)
    z = lambda x, y: gp3.r1(x, y) + gp3.r2(x, y)

    rz = z(x, y)

    eval_r1 = lambda x: gp3.r1(x[:, 0], x[:, 1])
    eval_r2 = lambda x: gp3.r2(x[:, 0], x[:, 1])
    eval_z = lambda x: z(x[:, 0], x[:, 1])

    pts = list(zip(x, y))
    train_x = np.array(pts)
    #train_f = rz
    # train_f = r1z
    train_f = r2z
    train_f=train_f.reshape((len(train_f),1))

    #------------------------------------------------------------
    k = lambda x,x_prime : gp3.matern_52(x,x_prime,1);  
    grad_k = lambda x,x_prime : grad_matern_52(x,x_prime,1); 
    #------------------------------------------------------------ 

    sampler = scipy.stats.qmc.LatinHypercube(2)
    test_pts = []

    for i in range(20):

        print("sample iteration number ", i+1)

        #------------------------------------------------------------
        def exp_imp(x):
            [m,s]=gp3.get_mean_and_sigma(x,train_x,train_f,k)
            return gp3.expected_improvement(train_f,m,s,0)
        grad_exp_imp = lambda x: grad_expected_improvement(x,train_x,train_f,k,grad_k)
        #------------------------------------------------------------

        x0= 8 * sampler.random(1) - 4
        t=0.1
        GApars=nesterovAscent(x0,grad_exp_imp,t,exp_imp)
        new_x=GApars[0]

        nx = (new_x[0,0], new_x[0,1])
        pts.append(nx)
        train_x = np.array(pts)
        #train_f = eval_z(train_x)
        #train_f = eval_r1(train_x)
        train_f = eval_r2(train_x)

        test_pts.append(nx)

    tp = np.array(test_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xbase = np.linspace(-4, 4, 100)
    ybase = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(xbase, ybase)
    # l1 = ax.contourf(X, Y, z(X, Y))
    # l1 = ax.contourf(X, Y, r1(X, Y))
    l1 = ax.contourf(X, Y, gp3.r2(X, Y))
    fig.colorbar(l1)
    ax.scatter(tp[:, 0], tp[:, 1], c='red')

    plt.show()

#------------------------------------------------------------
k = lambda x,x_prime : gp3.matern_52(x,x_prime,1);  
grad_k = lambda x,x_prime : grad_matern_52(x,x_prime,1); 
#------------------------------------------------------------ 

rng = np.random.default_rng(20230416)
x = 8 * rng.random(5) - 4
y = 8 * rng.random(5) - 4

pts = list(zip(x, y))
train_x=np.array(pts)
train_f = np.sum(train_x,axis=1)
train_f=train_f.reshape((len(train_f),1))

a=np.array([[1,1]]); b= np.array([[1,-1]]);

# Test that all these functions work with the desired size of inputs

g1=grad_matern_52(a,b,1)
g2=grad_matern_52(train_x,train_x,1); 

g3=grad_mu(a,train_x,train_f,k,grad_k)
g4=grad_mu(train_x+1,train_x,train_f,k,grad_k)

g5=grad_sigma(a,train_x,k,grad_k)
g6=grad_sigma(train_x+1,train_x,k,grad_k)

g7=grad_expected_improvement(a,train_x,train_f,k,grad_k)
g8=grad_expected_improvement(train_x+1,train_x,train_f,k,grad_k)

# print(g1); print(g2); print(g3); print(g4); print(g5); print(g6); print(g7); print(g8);

test_2d_withGD()