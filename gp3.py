import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

def cov_from_kernel(x, k):
    n = x.shape[0]
    cov = np.zeros((n, n))
    for i in range(n):
        xi = x[i]
        for j in range(i, n):
            xj = x[j]
            cov[i, j] = k(xi, xj)
    c1 = cov + cov.T
    c2 = c1 - np.eye(n)
    return c2

def cov_from_kernel1(x, y, k):
    n = x.shape[0]
    nstar = y.shape[0]
    cov = np.zeros((n, nstar))

    for i in range(n):
        xi = x[i]
        for j in range(nstar):
            yj = y[j]
            cov[i, j] = k(xi, yj)
    
    return cov

def matern_52(x, x_prime, l):
    """
    this will be matern assuming that nu = 5/2
    """
    if x.shape:
        r = np.linalg.norm(x - x_prime)
    else:
        r = np.abs(x - x_prime)
    c = 1 + np.sqrt(5)*r/l + (5*r**2)/(3*l**2)
    e = np.exp(-np.sqrt(5)*r/l)
    return c*e

def r1(x, y):
    return -1*x**2 - 3*y**2 - 4

def r2(x, y):
    return -2.5*x**2 - 1.5*y**2 - 8

def get_mean_and_sigma(test_x, train_x, train_f, k):
    k_x_x = cov_from_kernel(train_x, k)
    k_x_xstar = cov_from_kernel1(train_x, test_x, k)
    k_xstar_xstar = cov_from_kernel(test_x, k)

    kxx_inv = np.linalg.inv(k_x_x)

    m1 = k_x_xstar.T @kxx_inv @ train_f
    cov1 = k_x_xstar.T@kxx_inv@k_x_xstar
    cov = k_xstar_xstar - cov1

    var = np.diagonal(cov)
    sigma = np.sqrt(var)
    return m1, sigma

def expected_improvement(train_f, mu, sigma, xi=0.01):

    # Calculate the current best value f(x+) observed so far
    f_best = np.max(train_f)

    # Calculate the expected improvement
    with np.errstate(divide='ignore'):
        imp = mu - f_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def plot_ei(test_x, ei):
    
    xbase = np.linspace(-4, 4, 100)
    ybase = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(xbase, ybase)

    interp = CloughTocher2DInterpolator(test_x, ei)
    Z = interp(X, Y)

    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.plot(test_x[:, 0], test_x[:, 1], 'ok', label = 'input point')
    plt.legend()
    plt.colorbar()
    plt.show()


def test_2d():

    x = 8 * rng.random(20) - 4
    y = 8 * rng.random(20) - 4

    r1z = r1(x, y)
    r2z = r2(x, y)
    z = lambda x, y: r1(x, y) + r2(x, y)

    rz = z(x, y)

    eval_r1 = lambda x: r1(x[:, 0], x[:, 1])
    eval_r2 = lambda x: r2(x[:, 0], x[:, 1])
    eval_z = lambda x: z(x[:, 0], x[:, 1])

    pts = list(zip(x, y))
    train_x = np.array(pts)
    #train_f = rz
    # train_f = r1z
    train_f = r2z
    m52 = lambda x, y: matern_52(x, y, 1)

    sampler = scipy.stats.qmc.LatinHypercube(2)

    test_pts = []

    for i in range(10):
        test_x = 8 * sampler.random(25+i) - 4

        m1, sigma = get_mean_and_sigma(test_x, train_x, train_f, m52)

        ei = expected_improvement(train_f, m1, sigma)
        # plot_ei(test_x, ei)

        new_x = test_x[np.argmax(ei)]
        nx = (new_x[0], new_x[1])
        pts.append(nx)
        train_x = np.array(pts)
        #train_f = eval_z(train_x)
        #train_f = eval_r1(train_x)
        train_f = eval_r2(train_x)

        test_pts.append(nx)

    tp = np.array(test_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xbase = np.linspace(-10, 10, 100)
    ybase = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(xbase, ybase)
    # l1 = ax.contourf(X, Y, z(X, Y))
    # l1 = ax.contourf(X, Y, r1(X, Y))
    l1 = ax.contourf(X, Y, r2(X, Y))
    fig.colorbar(l1)
    ax.scatter(tp[:, 0], tp[:, 1], c='red')

    plt.show()


    




    # print('hello')

    # xbase = np.linspace(-4, 4, 100)
    # ybase = np.linspace(-4, 4, 100)
    # X, Y = np.meshgrid(xbase, ybase)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #ax.scatter(x, y)
    
    # l1 = ax.contourf(X, Y, z(X, Y))
    # fig.colorbar(l1)
    # ax.scatter(x, y, c='red')

    # plt.show()



rng = np.random.default_rng(20230416)
test_2d()