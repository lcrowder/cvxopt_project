"""
This script and associated functions are the first part of plotting gaussian processes. We define two covariance
from kernel functions which allow us to recover a covariance matrix given training points and a covariance function.

gen_funs() will generate some functions over [-5, 5] and plot them, with the covariance being k(x, y) = -1/2 |x - y|^2

pred from obs will use some points without noise (think observations) which will inform our posterior function draws
it then plots them along with the 2*sd around the mean line (this is the gray shading)
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio

def k_sqexp(x, y):
    return np.exp(-0.5 * np.abs(x - y)**2)

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

def gen_funs():
    n = 200
    t = np.linspace(-5, 5, n)
    c1 = cov_from_kernel(t, k_sqexp)

    num_fns = 10
    f_star = rng.multivariate_normal(np.zeros(n), c1, num_fns)
    print(f_star.shape)
    #t = np.linspace(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(num_fns):
        ax.plot(t, f_star[i, :], c='b', alpha=0.5)
    plt.show()

def make_gif():
    n = 200
    num_fns = 15
    train_x_all = np.array([-4, 3, -2, 1, 0])
    train_f_all = np.array([-2, .75, 1, 2, -1])

    test_x = np.linspace(-5, 5, n)
    k = lambda x, y : matern_52(x, y, 1)

    n_pts = len(train_x_all)

    k_xstar_xstar = cov_from_kernel(test_x, k)
    cov = k_xstar_xstar

    pw_mean=np.zeros(test_x.shape)
    
    sd_up = pw_mean + 2*np.sqrt(np.diagonal(cov))
    sd_down = pw_mean - 2*np.sqrt(np.diagonal(cov))
    
    # fig_test = plt.figure()
    # ax_test = fig_test.add_subplot()
    # ax_test.scatter(train_x, train_f)
    image_files = []

    y_lims = (-3.5, 3.6)
    x_lims = (-5.5, 5.5)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)

    ax.plot(test_x, pw_mean, c='r')
    ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='gray')
    ax.set_ylim(y_lims)
    ax.set_xlim(x_lims)

    num_fns = 3

    test_f = rng.multivariate_normal(pw_mean, cov, num_fns)

    for j in range(num_fns):
        ax.plot(test_x, test_f[j], c='b', alpha=0.15)

    #plt.show()
    image_files.append("gp2/{}.png".format('first'))
    fig1.savefig(image_files[-1], dpi=100, bbox_inches="tight")
    #plt.close()

    for i in range(n_pts):
        train_x = train_x_all[0:i+1]
        train_f = train_f_all[0:i+1]

        k_x_x = cov_from_kernel(train_x, k)
        k_x_xstar = cov_from_kernel1(train_x, test_x, k)
        k_xstar_xstar = cov_from_kernel(test_x, k)

        print("K_xx shape: {}".format(k_x_x.shape))
        print("K_x_xstar shape: {}".format(k_x_xstar.shape))
        print("K_xstar_xstar shape: {}".format(k_xstar_xstar.shape))

        m1 = k_x_xstar.T @ np.linalg.inv(k_x_x) @ train_f
        cov1 = k_x_xstar.T@np.linalg.inv(k_x_x)@k_x_xstar
        cov = k_xstar_xstar - cov1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(train_x, train_f, alpha=1, marker='+')
        ax.plot(test_x, m1, c='r')
        sd_up = m1 + 2*np.sqrt(np.diagonal(cov))
        sd_down = m1 - 2*np.sqrt(np.diagonal(cov))
        ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='gray')
        ax.set_ylim(y_lims)
        ax.set_xlim(x_lims)

        test_f = rng.multivariate_normal(m1, cov, num_fns)

        for j in range(num_fns):
            ax.plot(test_x, test_f[j], c='b', alpha=0.15)

        #plt.show()
        image_files.append("gp2/{}.png".format(i))
        fig.savefig(image_files[-1], dpi=100, bbox_inches="tight")
        #plt.close()
    
    filename = 'gif3.gif'
    duration = 2000
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(filename, images, duration=duration, loop=0)

def pred_from_obs():
    n = 200
    num_fns = 15
    train_x = np.array([-4, -3, -2, 0, 1])
    train_f = np.array([-2, 5, 1, 2, -1])

    test_x = np.linspace(-5, 5, n)
    k = lambda x, y : matern_52(x, y, 1)

    k_x_x = cov_from_kernel(train_x, k)
    k_x_xstar = cov_from_kernel1(train_x, test_x, k)
    k_xstar_xstar = cov_from_kernel(test_x, k)

    print("K_xx shape: {}".format(k_x_x.shape))
    print("K_x_xstar shape: {}".format(k_x_xstar.shape))
    print("K_xstar_xstar shape: {}".format(k_xstar_xstar.shape))

    m1 = k_x_xstar.T @ np.linalg.inv(k_x_x) @ train_f
    cov1 = k_x_xstar.T@np.linalg.inv(k_x_x)@k_x_xstar
    cov = k_xstar_xstar - cov1

    #print(np.diagonal(cov))

    test_f = rng.multivariate_normal(m1, cov, num_fns)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(train_x, train_f, alpha=1, marker='+')
    # for i in range(num_fns):
    #     ax.plot(test_x, test_f[i, :], c='b', alpha=0.15)

    #pw_mean = np.mean(test_f, axis=0)
    pw_mean = m1
    print(pw_mean.shape)

    ax.plot(test_x, pw_mean, c='r')
    sd_up = pw_mean + 2*np.sqrt(np.diagonal(cov))
    sd_down = pw_mean - 2*np.sqrt(np.diagonal(cov))

    ax.plot(test_x, sd_up, c='g')
    ax.plot(test_x, sd_down, c='b')
    #ax.fill_between(test_x, sd_up, sd_down, alpha=0.25, color='gray')
    print(ax.get_ylim())
    print(ax.get_xlim())
    plt.show()

def expected_improvement(test_x, train_x, train_f, mu, sigma, xi=0.01):

    # Calculate the current best value f(x+) observed so far
    f_best = np.max(train_f)

    # Calculate the expected improvement
    with np.errstate(divide='ignore'):
        imp = mu - f_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def obj(x):
    """
    our test objective function:
    f(x) = sin(x) + sin(10x/3)
    over domain (-6, 6) is multimodal and has global minimum at x=5.146
    """
    return np.sin(x) + np.sin((10./3.)*x)

def test_acq_fun():
    n = 200
    num_train = 10

    train_x = rng.uniform(-6, 6, size=num_train)
    train_f = obj(train_x)

    test_x = np.linspace(-6, 6, n)
    k = lambda x, y : matern_52(x, y, 1)

    k_x_x = cov_from_kernel(train_x, k)
    k_x_xstar = cov_from_kernel1(train_x, test_x, k)
    k_xstar_xstar = cov_from_kernel(test_x, k)

    k_x_x_inv = np.linalg.inv(k_x_x)

    m1 = k_x_xstar.T @k_x_x_inv @ train_f
    cov1 = k_x_xstar.T@k_x_x_inv@k_x_xstar
    cov = k_xstar_xstar - cov1

    var = np.diagonal(cov)
    sigma = np.sqrt(var)

    ei = expected_improvement(test_x, train_x, train_f, m1, sigma)

    next_sample = test_x[np.argmax(ei)]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    t = np.linspace(-6, 6, 1000)
    ax.plot(t, obj(t), label='Objective function')
    ax.scatter(train_x, train_f, marker='+', c='b')
    ax.scatter(next_sample, obj(next_sample), marker='x', c='r')

    ax2 = fig.add_subplot(122)
    ax2.plot(test_x, ei)
    

    plt.show()

def multiple_runs():
    num_test = 200
    num_train = 10

    train_x = rng.uniform(-6, 6, size=num_train)
    train_f = obj(train_x)

    test_x = np.linspace(-6, 6, num_test) ## maybe resample every iteration? For higher dimensions use LHS
    k = lambda x, y : matern_52(x, y, 1)

    num_iter = 35

    og_train_x = train_x
    new_samp_x = []

    for iter in range(num_iter):
        test_x = rng.uniform(-6, 6, size=num_test)
        k_x_x = cov_from_kernel(train_x, k)
        k_x_xstar = cov_from_kernel1(train_x, test_x, k)
        k_xstar_xstar = cov_from_kernel(test_x, k)

        m1 = k_x_xstar.T @ np.linalg.inv(k_x_x) @ train_f
        cov1 = k_x_xstar.T@np.linalg.inv(k_x_x)@k_x_xstar
        cov = k_xstar_xstar - cov1

        var = np.diagonal(cov)
        sigma = np.sqrt(var)

        ei = expected_improvement(test_x, train_x, train_f, m1, sigma)
        next_sample = test_x[np.argmax(ei)]

        train_x = np.append(train_x, next_sample)
        train_f = np.append(train_f, obj(next_sample))
        new_samp_x.append(next_sample)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.linspace(-6, 6, 1000)
    ax.plot(t, obj(t), label='Objective function')
    ax.scatter(og_train_x, obj(og_train_x), marker='+', c='b')
    ax.scatter(new_samp_x, obj(np.array(new_samp_x)), marker='x', c='r')

    #ax2 = fig.add_subplot(122)
    #ax2.plot(test_x, ei)

    print(train_x[np.argmax(train_f)], np.max(train_f))

    plt.show()



rng = default_rng(5050505)
make_gif()
#gen_funs()
#pred_from_obs()
#test_acq_fun()
#multiple_runs()

