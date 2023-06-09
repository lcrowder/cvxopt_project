import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import scipy
from scipy.stats import norm

import matplotlib.pyplot as plt
import imageio as iio
from matplotlib.transforms import Bbox

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

def plot_ei(test_x, ei, saveplot = False, fname='', train_x=[]):
    xbase = np.linspace(-4, 4, 100)
    ybase = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(xbase, ybase)

    interp = CloughTocher2DInterpolator(test_x, ei)
    Z = interp(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pmesh = ax.pcolormesh(X, Y, Z, shading='auto')#, vmin=0, vmax=1)
    ax.plot(test_x[:, 0], test_x[:, 1], 'ok', label = 'input point')
    if len(train_x) > 0:
        ax.scatter(train_x[:, 0], train_x[:, 1], c='r')
    #fig.colorbar(pmesh, ax=ax)
    if saveplot:
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        
    else:
        fig.colorbar(pmesh, ax=ax)
        plt.show()

    plt.close()

    # plt.pcolormesh(X, Y, Z, shading='auto')
    # plt.plot(test_x[:, 0], test_x[:, 1], 'ok', label = 'input point')
    # plt.legend()
    # plt.colorbar()
    # plt.show()

def test_2d(seed = 50):

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
    train_f_z = rz
    train_f_r1 = r1z
    train_f_r2 = r2z
    m52 = lambda x, y: matern_52(x, y, 1)

    sampler = scipy.stats.qmc.LatinHypercube(2, seed=seed)

    test_pts = []
    image_names=[]

    for i in range(20):
        test_x = 8 * sampler.random(100) - 4

        m1_z, sigma_z = get_mean_and_sigma(test_x, train_x, train_f_z, m52)
        # m1_r1, sigma_r1 = get_mean_and_sigma(test_x, train_x, train_f_r1, m52)
        # m1_r2, sigma_r2 = get_mean_and_sigma(test_x, train_x, train_f_r2, m52)

        eiz = expected_improvement(train_f_z, m1_z, sigma_z)
        # eir1 = expected_improvement(train_f_r1, m1_r1, sigma_r1)
        # eir2 = expected_improvement(train_f_r2, m1_r2, sigma_r2)
        
        fname = "ei/{}.png".format(i)
        image_names.append(fname)
        plot_ei(test_x, eiz, saveplot=True, fname = fname, train_x=train_x)
        # plot_ei(test_x, eir1)
        # plot_ei(test_x, eir2)

        new_x = test_x[np.argmax(eiz)]
        nx = (new_x[0], new_x[1])
        pts.append(nx)
        train_x = np.array(pts)
        train_f_z = eval_z(train_x)
        # train_f_r1 = eval_r1(train_x)
        # train_f_r2 = eval_r2(train_x)

        test_pts.append(nx)

    
    tp = np.array(test_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xbase = np.linspace(-4, 4, 100)
    ybase = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(xbase, ybase)
    l1 = ax.contourf(X, Y, z(X, Y))
    # l1 = ax.contourf(X, Y, r1(X, Y))
    #l1 = ax.contourf(X, Y, r2(X, Y))
    fig.colorbar(l1)
    ax.scatter(tp[:, 0], tp[:, 1], c='red')
    #ax.scatter(train_x[:, 0], train_x[:, 1], c='green')
    fig.savefig("ei/all.png", dpi=100)

    plt.show()

    images = []
    for f in image_names:
        images.append(iio.imread(f))
    iio.mimsave('ei_out.gif', images, duration = 750, loop=0)


    




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
test_2d(20230416)