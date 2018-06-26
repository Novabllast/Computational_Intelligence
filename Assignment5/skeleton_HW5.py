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
    # plot_iris_data(x_2dim, labels)

    # ------------------------
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm   

    # set parameters (only here)

    scenario = 1

    max_iter = 150
    nr_components = 3
    tol = 0.000000000001

    if scenario == 1:

        dim = 2

        # plot dataset before using EM and km_mean
        plot_iris_data(x_2dim, labels)
        plt.title('Sepal.Length / Petal.Length')
        plt.show()

        # algorithms
        alpha_0, mean_0, cov_0 = init_EM(x_2dim, dimension=dim, nr_components=nr_components, scenario=scenario)
        alpha_final, mean_final, cov_final, log_likelihood, labels_em = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0,
                                                                           max_iter, tol)
        centers_0 = init_k_means(x_2dim, dim, nr_components, scenario)
        centers, D, labels_k_mean = k_means(x_2dim, nr_components, centers_0, max_iter, tol)


        #############################
        #                           #
        #   VISULAIZATION IN 2-DIM  #
        #                           #
        #############################

        colors = ['r', 'g', 'b', 'y', 'c']

        xmin = np.min(x_2dim[:, 0])
        xmax = np.max(x_2dim[:, 0])
        ymin = np.min(x_2dim[:, 1])
        ymax = np.max(x_2dim[:, 1])

        # visualize k_mean results in 2-dim
        plot_iris_data(x_2dim, labels_k_mean)
        for k in range(0, nr_components):
            plt.scatter(centers[k, 0], centers[k, 1], s=1000, c='black', marker='+')
        plt.title('K_means Hard Classification (2dim)')
        plt.show()

        # visualize em results in 2-dim
        plot_iris_data(x_2dim, labels_em)
        for k in range(0, nr_components):
            plot_gauss_contour(mean_final[:, k], cov_final[:, :, k], xmin, xmax, ymin, ymax, len(data), 'EM Soft Classification (2dim)')
        plt.show()

        # plot standard dataset with em_labels
        plot_iris_data(x_2dim, labels_em)
        plt.title('Scatter plot soft classification (2-dim)')
        plt.show()

        # plot standard dataset with labels_k_mean
        plot_iris_data(x_2dim, labels_k_mean)
        plt.title('Scatter plot hard classification (2-dim)')
        plt.show()

    # ------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm
    elif scenario == 2:

        dim = 4

        # TODO set parameters
        tol = 0.0000000000001  # tolerance
        max_iter = 200  # maximum iterations for GN
        nr_components = 3  # n number of components

        # plot dataset before using EM and km_mean
        plot_iris_data(x_4dim, labels)
        plt.title('Sepal.Length / Sepal.Width')
        plt.show()

        # algorithms
        alpha_0, mean_0, cov_0 = init_EM(x_4dim, dimension=dim, nr_components=nr_components, scenario=scenario)
        alpha_final, mean_final, cov_final, LL, labels_em = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

        centers_0 = init_k_means(x_4dim, dim, nr_components, scenario)
        centers, D, labels_k_mean = k_means(x_4dim, nr_components, centers_0, max_iter, tol)

        #############################
        #                           #
        #   VISULAIZATION IN 4-DIM  #
        #                           #
        #############################

        xmin = np.min(x_4dim[:, 0])
        xmax = np.max(x_4dim[:, 0])
        ymin = np.min(x_4dim[:, 2])
        ymax = np.max(x_4dim[:, 2])

        # visualize k_mean results in 4-dim
        plot_iris_data(x_4dim[:, [0, 2]], labels_k_mean)
        for k in range(0, nr_components):
            plt.scatter(centers[k, 0], centers[k, 2], s=1000, c='black', marker='+')
        plt.title('K_means Hard Classification (4dim)')
        plt.show()

        # visualize em results in 4-dim
        plot_iris_data(x_4dim[:, [0, 2]], labels_em)
        for k in range(0, nr_components):
            plot_gauss_contour(mean_final[[0, 2], k], cov_final[:, :, k], xmin, xmax, ymin, ymax, len(data),
                               'EM Soft Classification (4dim)')
        plt.show()

        # plot standard dataset with em_labels
        plot_iris_data(x_4dim[:, [0, 2]], labels_em)
        plt.title('Scatter plot soft classification (4-dim)')
        plt.show()

        # plot standard dataset with labels_k_mean
        plot_iris_data(x_4dim[:, [0, 2]], labels_k_mean)
        plt.title('Scatter plot hard classification (4-dim)')
        plt.show()

    # ------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data

    elif scenario == 3:

        dim = 2

        # plot dataset before using EM and km_mean
        plot_iris_data(x_2dim_pca[0], labels)
        plt.title('Data set PCA')
        plt.show()

        # algorithms
        alpha_0, mean_0, cov_0 = init_EM(x_2dim_pca[0], dimension=dim, nr_components=nr_components, scenario=scenario)
        alpha_final, mean_final, cov_final, LL, labels_em = EM(x_2dim_pca[0], nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

        centers_0 = init_k_means(x_2dim_pca[0], dim, nr_components, scenario)
        centers, D, labels_k_mean = k_means(x_2dim_pca[0], nr_components, centers_0, max_iter, tol)

        #############################
        #                           #
        #   VISULAIZATION OF PCA    #
        #                           #
        #############################

        xmin = np.min(x_2dim_pca[0][:, 0])
        xmax = np.max(x_2dim_pca[0][:, 0])
        ymin = np.min(x_2dim_pca[0][:, 1])
        ymax = np.max(x_2dim_pca[0][:, 1])

        # visualize k_mean results in 2-dim
        plot_iris_data(x_2dim_pca[0], labels_k_mean)
        for k in range(0, nr_components):
            plt.scatter(centers[k, 0], centers[k, 1], s=1000, c='black', marker='+')
        plt.title('K_means Hard Classification (PCA)')
        plt.show()

        # visualize em results in 2-dim
        plot_iris_data(x_2dim_pca[0], labels_em)
        for k in range(0, nr_components):
            plot_gauss_contour(mean_final[:, k], cov_final[:, :, k], xmin, xmax, ymin, ymax, len(data),
                               'EM Soft Classification (PCA)')
        plt.show()

        # plot standard dataset with em_labels
        plot_iris_data(x_2dim_pca[0], labels_em)
        plt.title('Scatter plot soft classification (PCA)')
        plt.show()

        # plot standard dataset with labels_k_mean
        plot_iris_data(x_2dim_pca[0], labels_k_mean)
        plt.title('Scatter plot hard classification (PCA)')
        plt.show()

    # TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)


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

    for k in range(0, nr_components):
        mean_0[:, k] = data[random.randint(0, len(data) - 1)]
        cov_0[:, :, k] = np.cov(data, rowvar=False)

    return alpha_0, mean_0, cov_0


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

    N = len(X)
    r_kn = np.zeros((K, N))

    labels = np.zeros(N)

    LL_prev = 0.0
    LL_array = list()

    for i in range(0, max_iter):

        r_sum = 0

        # expectation step
        for k in range(0, K):
            likelihood = likelihood_multivariate_normal(X, mean_0[:, k], cov_0[:, :, k])
            # calculate r_sum
            r_sum += likelihood * alpha_0[k]

        for k in range(0, K):
            likelihood = likelihood_multivariate_normal(X, mean_0[:, k], cov_0[:, :, k])
            r_kn[k, :] = likelihood * alpha_0[k] / r_sum

        # maximization step
        for k in range(0, K):

            N_k = np.sum(r_kn[k, :])

            # update mean_0
            mean_0[:, k] = np.average(X, axis=0, weights=r_kn[k, :])

            # update alpha_0
            alpha_0[k] = N_k / N

            # update cov_0
            cov_0[:, :, k] = np.cov(X, rowvar=False, ddof=0, aweights=r_kn[k, :])


        tmp_likelihood = sum(np.log(r_sum))
        LL_array.append(tmp_likelihood)

        # Check wether the likelihood has already converged or not
        if (np.abs(LL_prev - tmp_likelihood) < tol):
            break

        LL_prev = tmp_likelihood

    plt.show()
    plt.plot(LL_array)
    plt.title("log likelihood curve (EM algorithm) over iterations")
    plt.show()

    # get updated labels
    for i, label in enumerate(np.argmax(r_kn, axis=0)):
        labels[i] = reassign_class_labels(np.argmax(r_kn, axis=0))[label]

    return alpha_0, mean_0, cov_0, LL_array, labels


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

    initial_centers = np.empty((nr_clusters, dimension))
    for m in range(0, nr_clusters):
        initial_centers[m, :] = random.choice(dataset)

    # test data
    # initial_centers[0, :] = [6., 4.25]
    # initial_centers[1, :] = [6.2, 4.8]
    # initial_centers[2, :] = [5.6, 3.85]
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

    j_prev = 0
    cumulative_distance = []
    centers_backup = np.zeros(centers_0.shape)

    y = np.zeros(X[:, 1].size)

    for iteration in range(0, max_iter):

        # Step 1: Klassifikation der Samples zu den Komponenten (→ modifizierter E-step)
        distance = cdist(X, centers_0, metric="euclidean")
        y = np.argmin(distance, axis=1)

        # Step 2: Neuberechnung der Mittelwertvektoren (entspricht Schwerpunkt der Cluster) aufgrund der Zuweisung in Yk
        for cluster in range(0, K):
            sum_x = np.sum(X[np.where(y == cluster), :], axis=1)
            y_k = X[np.where(y == cluster), :].shape[1]
            centers_0[cluster, :] = sum_x / y_k

        # 4. Evaluieren der kumulativen Distanz
        j = 0
        for cluster in range(K):
            j += pow((X[np.where(y == cluster), :] - centers_0[cluster, :]), 2).sum()

        centers_backup[:] = centers_0
        cumulative_distance.append(j)

        if abs(j - j_prev) < tol:
            break

        j_prev = j

    plt.plot(cumulative_distance)
    plt.title('Cumulative distance')
    plt.show()

    return centers_0, cumulative_distance, y


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

    dim = 0
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # using sklearn pca (just for comparison)
    #sklearn_pca = sklearnPCA(n_components=dim)
    #transformed_by_sklearn = sklearn_pca.fit_transform(data)

    #print('Transformed SKLEAR: \n', transformed_by_sklearn)

    # TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components
    # https://plot.ly/ipython-notebooks/principal-component-analysis/

    # another thought on cov_mat (different results!!!)
    mean_vec = np.mean(data, axis=0)
    cov_mat1 = (data - mean_vec).T.dot((data - mean_vec)) / (data.shape[0] - 1)

    cov_mat = np.cov(data.T)


    eigen_values, eigen_vectors = np.linalg.eig(cov_mat1)

    #print('NumPy covariance matrix 1: \n', cov_mat1)
    #print('NumPy covariance matrix 2: \n', cov_mat2)
    #print('Eigenvectors \n', eigen_vectors)
    #print('Eigenvalues \n', eigen_values)

    ###############################################################################################

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    #print('Eigenvalues in descending order\n', eigen_values)

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                          eig_pairs[1][1].reshape(4, 1)))

    #print('Matrix W:\n', matrix_w)

    # for ev in eigen_vectors:
    #     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    ###############################################################################################

    # TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained

    # λ_1 = u^T_1 Su_1 = σ^2
    variance_explained = [(i / sum(eigen_values)) * 100 for i in sorted(eigen_values, reverse=True)]

    # y_n = U^T x_n
    #transformed = np.dot(eigen_vectors[:dim], data.T)
    transformed = data.dot(matrix_w)

    # yn =L ^{− 1/ 2} U^T (xn − mx)
    if whitening:
        # https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
        # a fudge factor can be used so that eigenvectors associated with
        # small eigenvalues do not get overamplified.
        D = np.diag(1. / np.sqrt(eigen_values + 1E-18))

        # whitening matrix
        W = np.dot(np.dot(eigen_vectors, D), eigen_vectors.T)

        # multiply by the whitening matrix
        X_white = np.dot(data, W)

    #print('Transformed: \n', transformed)

    #return transformed_by_sklearn, variance_explained
    return transformed, variance_explained


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
    plt.show()

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
