# Filename: HW5_skeleton.py
# Author: Christian Knoll
# Edited: May, 2018
import random

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.spatial.distance import cdist

from scipy.stats import multivariate_normal
import pdb

import sklearn
from sklearn import datasets


# --------------------------------------------------------------------------------
# Assignment 5
def main():
    # ------------------------
    # 0) Get the input 
    # (a) load the modified iris data
    data, labels = load_iris_data()

    # (b) construct the datasets
    x_2dim = data[:, [0, 2]]
    x_4dim = data
    # TODO: implement PCA
    x_2dim_pca = PCA(data, nr_dimensions=2, whitening=False)

    # (c) visually inspect the data with the provided function (see example below)
    plot_iris_data(x_2dim, labels)

    # ------------------------
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm   
    scenario = 1
    dim = 2
    nr_components = 3

    # TODO set parameters
    tol = 0.00000000001  # tolerance
    max_iter = 300  # maximum iterations for GN (maybe this is the number of N = 150)
    # nr_components = ... #n number of components

    # TODO: implement
    alpha_0, mean_0, cov_0 = init_EM(x_2dim, dimension=dim, nr_components=nr_components, scenario=scenario)
    alpha_final, mean_final, cov_final, log_likelihood, labels_em = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0,
                                                                       max_iter, tol)
    centers_0 = init_k_means(x_2dim, dim, nr_components, scenario)
    centers, D, labels_k_mean = k_means(x_2dim, nr_components, centers_0, max_iter, tol)

    # TODO visualize your results
    # labels = reassign_class_labels(labels)
    plot_iris_data(data, labels_em)
    plot_iris_data(data, labels_k_mean)

    colors = ['r', 'g', 'b', 'y', 'c']

    # for i in range(0, nr_components):
    #     points = centers[i]
    #     plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    # plt.scatter(centers_0[:, 0], centers_0[:, 1], marker='*', s=200, c='#050505')
    #
    # plt.show()

    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    for k in range(0, nr_components):
        plt.scatter(centers[k, 0], centers[k, 1], s=7, c=colors[k], marker="*")
        plot_gauss_contour(mean_final[:, k], cov_final[:, k], xmin, xmax, ymin, ymax, len(data), 'plot_gauss_contour')

    plt.show()

    # ------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm 
    scenario = 2
    dim = 4
    nr_components = 3

    # TODO set parameters
    tol = 0.0000000000001  # tolerance
    max_iter = 200  # maximum iterations for GN
    nr_components = 3  # n number of components

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_4dim, dimension=dim, nr_components=nr_components, scenario=scenario)
    alpha, mu, cov, LL, labels_em = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

    centers_0 = init_k_means(x_4dim, dim, nr_components, scenario)
    centers, D, labels_k_mean = k_means(x_4dim, nr_components, centers_0, max_iter, tol)

    # TODO: visualize your results by looking at the same slice as in 1)
    plot_iris_data(data, labels_em)
    plot_iris_data(data, labels_k_mean)

    xmin = np.min(x_4dim[:, 0])
    xmax = np.max(x_4dim[:, 0])
    ymin = np.min(x_4dim[:, 1])
    ymax = np.max(x_4dim[:, 1])
    for k in range(0, nr_components):
        plt.scatter(centers[k, 0], centers[k, 1], s=7, c=colors[k], marker="*")
        plot_gauss_contour(mean_final[:, k], cov_final[:, k], xmin, xmax, ymin, ymax, len(data), 'plot_gauss_contour')

    plt.show()

    # ------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 3
    # TODO set parameters
    tol = 0.00000000001  # tolerance
    max_iter = 300  # maximum iterations for GN
    nr_components = 3  # n number of components

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_2dim, dimension=dim, nr_components=nr_components, scenario=scenario)
    alpha, mu, cov, LL, labels_em = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

    centers_0 = init_k_means(x_2dim, dim, nr_components, scenario)
    centers, D, labels_k_mean = k_means(x_2dim, nr_components, centers_0, max_iter, tol)

    # TODO: visualize your results
    plot_iris_data(data, labels_em)
    plot_iris_data(data, labels_k_mean)

    xmin = np.min(x_2dim[:, 0])
    xmax = np.max(x_2dim[:, 0])
    ymin = np.min(x_2dim[:, 1])
    ymax = np.max(x_2dim[:, 1])
    for k in range(0, nr_components):
        plt.scatter(centers[k, 0], centers[k, 1], s=7, c=colors[k], marker="*")
        plot_gauss_contour(mean_final[:, k], cov_final[:, k], xmin, xmax, ymin, ymax, len(data), 'plot_gauss_contour')

    plt.show()

    # TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)

    pdb.set_trace()
    pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def init_EM(data, dimension=2, nr_components=3, scenario=None):
    """ initializes the EM algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""
    # TODO choose suitable initial values for each scenario

    # 1.) EM algorithm for GMM:

    # initializing alpha - uniforme Verteilungsfunktion
    alpha_0 = np.ones((nr_components,)) * (1 / nr_components)

    # initializing mean - select m samples randomly
    mean_0 = np.ones((dimension, nr_components))

    # initializing cov_0 dimension
    cov_0 = np.zeros((dimension, dimension, nr_components))

    for i in range(0, nr_components):
        mean_0[:, i] = data[random.randint(0, len(data) - 1)]
        cov_0[:, :, i] = np.cov(data, rowvar=False)

    return alpha_0, mean_0, cov_0

    pass


# --------------------------------------------------------------------------------
def EM(X, K, alpha_0, mean_0, cov_0, max_iter, tol):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K        
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension 
    D = X.shape[1]
    assert D == mean_0.shape[0]
    # TODO: iteratively compute the posterior and update the parameters

    classify = True

    N = len(X)
    r_kn = np.zeros((K, N))
    # r_kn = np.zeros((N, K))

    labels = np.zeros(K)

    L = 0.0
    L_old = 0.0
    L_array = np.zeros((max_iter))

    for i in range(0, max_iter):

        # expectation step
        for k in range(0, K):
            likelihood = likelihood_multivariate_normal(X, mean_0[:, k], cov_0[:, :, k])
            r_kn[k, :] = likelihood * alpha_0[k]

        # calculate sum
        r_kn_sum = np.sum(r_kn, axis=0)

        r_kn = np.einsum('mn, n->mn', r_kn, np.reciprocal(r_kn_sum))

        # maximization step
        for k in range(0, K):
            N_k = np.sum(r_kn, axis=1)

            # update mean_0
            mean_0[:, k] = np.average(X, axis=0, weights=r_kn[k, :])
            # mean_0[:, k] = (1 / N_k) * (np.sum(r_kn, axis=1) * np.sum(X[i, :], axis=1))

            # update alpha_0
            # alpha_0[k] = N_k / max_iter

            alpha_0[k] = np.sum(r_kn[k, :]) / N

            # update cov_0
            # cov_0[:, :, k] = (1 / N_k) * (np.sum(r_kn, axis=1) * np.sum(X[i, :] - mean_0[:, k], axis=1)
            #                               * np.sum((X[i, :] - mean_0[:, k]), axis=0))

            cov_0[:, :, k] = np.cov(X, rowvar=False, ddof=0, aweights=r_kn[k, :])

        # likelihood calculation
        L = sum(np.log(r_kn_sum))

        L_array[i] = L
        if (np.abs(L_old - L) < tol):
            plt.plot(L_array[:i])
            plt.show()
            if (classify == True):
                y = np.argmax(r_kn, axis=0)
                plt.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
                plt.title('Classification of EM')
                plt.xlabel('x')
                plt.ylabel(('y'))
                plt.show()
            return alpha_0, mean_0, cov_0, L, labels
        L_old = L

        # if diag == True:
        #     Sigma[m, :, :] = np.diag(np.diag(Sigma[m, :, :]))

        #           rsum1 = np.sum(r, axis=1)
        # mean_0[:, k] = np.average(X, axis=0, weights=r_kn[:, k])
        #
        #         alpha[m] = sum(r[m, :]) / N
        #
        #         Sigma[m, :, :] = np.cov(X, rowvar=False, ddof=0, aweights=r[m, :])
        #         if diag == True:
        #             Sigma[m, :, :] = np.diag(np.diag(Sigma[m, :, :]))

        # likelihood calculation
        # L = sum(np.log(r_kn_sum))
        # L_array[i] = L
        # print("L", L)
        # if (np.abs(L_old - L) < 10 ** -12):
        #     plt.plot(L_array[:i])
        #     plt.show()
        #     if (classify == True):
        #         y = np.argmax(r_kn, axis=0)
        #         plt.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
        #         plt.title('Classification of EM')
        #         plt.xlabel('x')
        #         plt.ylabel(('y'))
        #         plt.show()
        #     return alpha_0, mean_0, cov_0, L
        # L_old = L

    plt.plot(L_array)
    plt.show()
    if (classify == True):
        y = np.argmax(r_kn, axis=0)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
        plt.title('Classification of EM')
        plt.xlabel('x')
        plt.ylabel(('y'))
        plt.show()

    return alpha_0, mean_0, cov_0, L, labels

    # for m in range(0, M):
    # rsum1 = np.sum(r, axis=1)
    #         mu[m, :] = np.average(X, axis=0, weights=r[m, :])
    #
    #         alpha[m] = sum(r[m, :]) / N
    #
    #         Sigma[m, :, :] = np.cov(X, rowvar=False, ddof=0, aweights=r[m, :])
    #         if diag == True:
    #             Sigma[m, :, :] = np.diag(np.diag(Sigma[m, :, :]))

    # print("started EM algorithm")
    # N = len(X)  # X.size/2
    # print(X.size, X.shape, len(X))
    # alpha = np.ones((M,)) * alpha_0
    # mu = mu_0
    # Sigma = np.ones((M, 2, 2))
    # r = np.zeros((M, N))
    # L = 0.0
    # L_old = 0.0
    # L_array = np.zeros((max_iter,))
    #
    # print("going to assign sigma_0")
    # for m in range(0, M):
    #     Sigma[m, :, :] *= Sigma_0
    #
    # # print(Sigma)
    #
    # plt.xlabel('Iterations')
    # plt.ylabel('Log Likelihood')
    # plt.title('Log-Likelihood over the Iterations')
    #
    # for i in range(0, max_iter):
    #
    #     if (i % 10 == 0):
    #         print("iteration", i)
    #
    #     # expectation step
    #     for m in range(0, M):
    #         dist = multivariate_normal(mu[m, :], Sigma[m, :, :])
    #         r[m, :] = dist.pdf(X) * alpha[m]
    #
    #     rsum = np.sum(r, axis=0)
    #
    #     r = np.einsum('mn, n->mn', r, np.reciprocal(rsum))
    #
    #     # likelihood calculation
    #     L = sum(np.log(rsum))
    #     L_array[i] = L
    #     # print("L", L)
    #     if (np.abs(L_old - L) < 10 ** -12):
    #         plt.plot(L_array[:i])
    #         plt.show()
    #         if (classify == True):
    #             y = np.argmax(r, axis=0)
    #             plt.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
    #             plt.title('Classification of EM')
    #             plt.xlabel('x')
    #             plt.ylabel(('y'))
    #             plt.show()
    #         return alpha, mu, Sigma, L
    #     L_old = L
    #
    #     # maximization step
    #     for m in range(0, M):
    #         rsum1 = np.sum(r, axis=1)
    #         mu[m, :] = np.average(X, axis=0, weights=r[m, :])
    #
    #         alpha[m] = sum(r[m, :]) / N
    #
    #         Sigma[m, :, :] = np.cov(X, rowvar=False, ddof=0, aweights=r[m, :])
    #         if diag == True:
    #             Sigma[m, :, :] = np.diag(np.diag(Sigma[m, :, :]))
    #
    # plt.plot(L_array)
    # plt.show()
    # if (classify == True):
    #     y = np.argmax(r, axis=0)
    #     plt.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
    #     plt.title('Classification of EM')
    #     plt.xlabel('x')
    #     plt.ylabel(('y'))
    #     plt.show()

    # return alpha, mu, Sigma, L

    # TODO: classify all samples after convergence
    pass


# --------------------------------------------------------------------------------
def init_k_means(dataset, dimension=None, nr_clusters=None, scenario=None):
    """ initializes the k_means algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters
        :param dataset: """
    # TODO choose suitable initial values for each scenario

    initial_centers = np.empty((nr_clusters, 2))
    for m in range(0, nr_clusters):
        initial_centers[m, :] = random.choice(dataset)

    return initial_centers


# --------------------------------------------------------------------------------
def k_means(X, K, centers_0, max_iter, tol):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    # assert D == centers_0.shape[0] # WTH What's the purpose of that
    assert D == centers_0.shape[1]
    # TODO: iteratively update the cluster centers

    j_prev = 0

    centers = {0: np.zeros(X.shape), 1: np.zeros(X.shape), 2: np.zeros(X.shape)}
    centers_backup = np.zeros(centers_0.shape)
    distances = np.zeros([X.shape[0], K])
    cumulative_distance = 0
    j = 0
    for iteration in range(0, max_iter):
        # Step 1: Klassifikation der Samples zu den Komponenten (→ modifizierter E-step)
        for cluster in range(0, K):
            distance = pow((X - centers_0[cluster]), 2).sum(axis=1)
            distances[:, cluster] = distance

        y = np.argmin(distances, axis=1)

        distance = cdist(X, centers_0, metric="euclidean")
        g = np.argmin(distance, axis=1)

        # Step 2: Neuberechnung der Mittelwertvektoren (entspricht Schwerpunkt der Cluster) aufgrund der Zuweisung in Yk
        for cluster in range(0, K):
            sum_x = np.sum(X[np.where(y == cluster), :], axis=1)
            y_k = X[np.where(y == cluster), :].shape[1]
            centers_0[cluster, :] = sum_x / y_k

        # 4. Evaluieren der kumulativen Distanz
        for k in range(K):
            j += np.sum(np.linalg.norm(X[y == k] - centers_backup[k]))

        centers_backup[:] = centers_0
        cumulative_distance += j

        if abs(j - j_prev) < tol:
            break

        j_prev = j

    # TODO: classify all samples after convergence

    return centers_0, cumulative_distance, j


# --------------------------------------------------------------------------------
def PCA(data, nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening
        
    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components

    # TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input: 
    Returns:
        X... samples, 150x4
        Y... labels, 150x1"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100, 2] = iris.data[50:100, 2] - 0.25
    Y = iris.target
    return X, Y


# --------------------------------------------------------------------------------
def plot_iris_data(data, labels):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1"""
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1], label='Iris-Setosa')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Iris-Versicolor')
    plt.scatter(data[labels == 2, 0], data[labels == 2, 1], label='Iris-Virgnica')

    plt.legend()
    plt.show()


# --------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
    """Returns the likelihood of X for multivariate (d-dimensional) Gaussian
    specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
    mean ... mean -- [mu_1, mu_2,...,mu_d]
    cov ... covariance matrix -- np.array with (d x d)
    log ... False for likelihood, true for log-likelihood
    """

    dist = multivariate_normal(mean, cov)
    if log is False:
        P = dist.pdf(X)
    elif log is True:
        P = dist.logpdf(X)
    return P


# --------------------------------------------------------------------------------
def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, nr_points, title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

    # npts = 100
    delta_x = float(xmax - xmin) / float(nr_points)
    delta_y = float(ymax - ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)

    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, np.sqrt(cov[0][0]), np.sqrt(cov[1][1]), mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return


# --------------------------------------------------------------------------------
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.
       
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)

    y = np.zeros(N)
    cumulativePM = np.cumsum(PM)  # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N)  # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N)  # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]:  # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y)  # permutation of all samples


# --------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable. 
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50] == 0), np.sum(labels[0:50] == 1), np.sum(labels[0:50] == 2)],
                                  [np.sum(labels[50:100] == 0), np.sum(labels[50:100] == 1),
                                   np.sum(labels[50:100] == 2)],
                                  [np.sum(labels[100:150] == 0), np.sum(labels[100:150] == 1),
                                   np.sum(labels[100:150] == 2)]])
    new_labels = np.array([np.argmax(class_assignments[:, 0]),
                           np.argmax(class_assignments[:, 1]),
                           np.argmax(class_assignments[:, 2])])
    return new_labels


# --------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu = [0.0, 0.0]
    cov = [[1, 0.2], [0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    plot_gauss_contour(mu, cov, -2, 2, -2, 2, 100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)

    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))

    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                                        0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                                        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels = np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd == 0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd == 1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd == 2] = new_labels[2]


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    sanity_checks()
    main()
