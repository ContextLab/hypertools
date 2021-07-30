# copied from: https://github.com/brainiak/brainiak/blob/master/brainiak/funcalign/srm.py
#              https://github.com/brainiak/brainiak/blob/master/brainiak/funcalign/rsrm.py

# coding: latin-1

#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Shared Response Model (SRM)

The implementations are based on the following publications:

.. [Chen2015] "A Reduced-Dimension fMRI Shared Response Model",
   P.-H. Chen, J. Chen, Y. Yeshurun-Dishon, U. Hasson, J. Haxby, P. Ramadge
   Advances in Neural Information Processing Systems (NIPS), 2015.
   http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model

.. [Anderson2016] "Enabling Factor Analysis on Thousand-Subject Neuroimaging
   Datasets",
   Michael J. Anderson, Mihai CapotÄ, Javier S. Turek, Xia Zhu, Theodore L.
   Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning, Peter J. Ramadge,
   Kenneth A. Norman, arXiv preprint, 2016.
   https://arxiv.org/abs/1608.04647

.. [Turek2017] "Capturing Shared and Individual Information in fMRI Data",
   J. Turek, C. Ellis, L. Skalaban, N. Turk-Browne, T. Willke
   under review, 2017.
"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek
# (Intel Labs), 2015
from __future__ import division

import logging

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError
import uuid
import hashlib


__all__ = [
    "SRM", "DetSRM", "RSRM"
]

logger = logging.getLogger(__name__)


def _init_w_transforms(data, features):
    """Initialize the mappings (Wi) for the SRM with random orthogonal matrices.

    Parameters
    ----------

    data : list of 2D arrays, element i has shape=[voxels_i, samples]
        Each element in the list contains the fMRI data of one subject.

    features : int
        The number of features in the model.


    Returns
    -------

    w : list of array, element i has shape=[voxels_i, features]
        The initialized orthogonal transforms (mappings) :math:`W_i` for each
        subject.

    voxels : list of int
        A list with the number of voxels per subject.


    Note
    ----

        This function assumes that the numpy random number generator was
        initialized.

        Not thread safe.
    """
    w = []
    subjects = len(data)
    voxels = np.empty(subjects, dtype=int)

    # Set Wi to a random orthogonal voxels by features matrix
    for subject in range(subjects):
        voxels[subject] = data[subject].shape[0]
        rnd_matrix = np.random.random((voxels[subject], features))
        q, r = np.linalg.qr(rnd_matrix)
        w.append(q)

    return w, voxels


class SRM(BaseEstimator, TransformerMixin):
    """Probabilistic Shared Response Model (SRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    sigma_s_ : array, shape=[features, features]
        The covariance of the shared response Normal distribution.

    mu_ : list of array, element i has shape=[voxels_i]
        The voxel means over the samples for each subject.

    rho2_ : array, shape=[subjects]
        The estimated noise variance :math:`\\rho_i^2` for each subject


    Note
    ----

       The number of voxels may be different between subjects. However, the
       number of samples must be the same across subjects.

       The probabilistic Shared Response Model is approximated using the
       Expectation Maximization (EM) algorithm proposed in [Chen2015]_. The
       implementation follows the optimizations published in [Anderson2016]_.

       This is a single node version.

       The run-time complexity is :math:`O(I (V T K + V K^2 + K^3))` and the
       memory complexity is :math:`O(V T)` with I - the number of iterations,
       V - the sum of voxels from all subjects, T - the number of samples, and
       K - the number of features (typically, :math:`V \\gg T \\gg K`).
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(self, X, y=None):
        """Compute the probabilistic Shared Response Model

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used
        """
        logger.info('Starting Probabilistic SRM')

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects "
                             "({0:d}) to train the model.".format(len(X)))

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects"
                                 ".")

        # Run SRM
        self.sigma_s_, self.w_, self.mu_, self.rho2_, self.s_ = self._srm(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform matrix to Shared Response space

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject
            note that number of voxels and samples can vary across subjects
        y : not used (as it is unsupervised learning)


        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _init_structures(self, data, subjects):
        """Initializes data structures for SRM and preprocess the data.


        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        subjects : int
            The total number of subjects in `data`.


        Returns
        -------
        x : list of array, element i has shape=[voxels_i, samples]
            Demeaned data for each subject.

        mu : list of array, element i has shape=[voxels_i]
            Voxel means over samples, per subject.

        rho2 : array, shape=[subjects]
            Noise variance :math:`\\rho^2` per subject.

        trace_xtx : array, shape=[subjects]
            The squared Frobenius norm of the demeaned data in `x`.
        """
        x = []
        mu = []
        rho2 = np.zeros(subjects)

        trace_xtx = np.zeros(subjects)
        for subject in range(subjects):
            mu.append(np.mean(data[subject], 1))
            rho2[subject] = 1
            trace_xtx[subject] = np.sum(data[subject] ** 2)
            x.append(data[subject] - mu[subject][:, np.newaxis])

        return x, mu, rho2, trace_xtx

    def _likelihood(self, chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
                    trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
                    samples):
        """Calculate the log-likelihood function


        Parameters
        ----------

        chol_sigma_s_rhos : array, shape=[features, features]
            Cholesky factorization of the matrix (Sigma_S + sum_i(1/rho_i^2)
            * I)

        log_det_psi : float
            Determinant of diagonal matrix Psi (containing the rho_i^2 value
            voxels_i times).

        chol_sigma_s : array, shape=[features, features]
            Cholesky factorization of the matrix Sigma_S

        trace_xt_invsigma2_x : float
            Trace of :math:`\\sum_i (||X_i||_F^2/\\rho_i^2)`

        inv_sigma_s_rhos : array, shape=[features, features]
            Inverse of :math:`(\\Sigma_S + \\sum_i(1/\\rho_i^2) * I)`

        wt_invpsi_x : array, shape=[features, samples]

        samples : int
            The total number of samples in the data.


        Returns
        -------

        loglikehood : float
            The log-likelihood value.
        """
        log_det = (np.log(np.diag(chol_sigma_s_rhos) ** 2).sum() + log_det_psi
                   + np.log(np.diag(chol_sigma_s) ** 2).sum())
        loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
        loglikehood += 0.5 * np.trace(
            wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x))
        # + const --> -0.5*nTR*nvoxel*subjects*math.log(2*math.pi)

        return loglikehood

    def _srm(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.


        Returns
        -------

        sigma_s : array, shape=[features, features]
            The covariance :math:`\\Sigma_s` of the shared response Normal
            distribution.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        mu : list of array, element i has shape=[voxels_i]
            The voxel means :math:`\\mu_i` over the samples for each subject.

        rho2 : array, shape=[subjects]
            The estimated noise variance :math:`\\rho_i^2` for each subject

        s : array, shape=[features, samples]
            The shared response.
        """

        samples = data[0].shape[1]
        subjects = len(data)

        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject, and trace_xtx with
        # the ||X_i||_F^2 of each subject.
        w, voxels = _init_w_transforms(data, self.features)
        x, mu, rho2, trace_xtx = self._init_structures(data, subjects)
        shared_response = np.zeros((self.features, samples))
        sigma_s = np.identity(self.features)

        # Main loop of the algorithm (run
        for iteration in range(self.n_iter):
            logger.info('Iteration %d' % (iteration + 1))

            # E-step:

            # Sum the inverted the rho2 elements for computing W^T * Psi^-1 * W
            rho0 = (1 / rho2).sum()

            # Invert Sigma_s using Cholesky factorization
            (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(
                sigma_s, check_finite=False)
            inv_sigma_s = scipy.linalg.cho_solve(
                (chol_sigma_s, lower_sigma_s), np.identity(self.features),
                check_finite=False)

            # Invert (Sigma_s + rho_0 * I) using Cholesky factorization
            sigma_s_rhos = inv_sigma_s + np.identity(self.features) * rho0
            (chol_sigma_s_rhos, lower_sigma_s_rhos) = scipy.linalg.cho_factor(
                sigma_s_rhos, check_finite=False)
            inv_sigma_s_rhos = scipy.linalg.cho_solve(
                (chol_sigma_s_rhos, lower_sigma_s_rhos),
                np.identity(self.features), check_finite=False)

            # Compute the sum of W_i^T * rho_i^-2 * X_i, and the sum of traces
            # of X_i^T * rho_i^-2 * X_i
            wt_invpsi_x = np.zeros((self.features, samples))
            trace_xt_invsigma2_x = 0.0
            for subject in range(subjects):
                wt_invpsi_x += (w[subject].T.dot(x[subject])) / rho2[subject]
                trace_xt_invsigma2_x += trace_xtx[subject] / rho2[subject]

            log_det_psi = np.sum(np.log(rho2) * voxels)

            # Update the shared response
            shared_response = sigma_s.dot(
                np.identity(self.features) - rho0 * inv_sigma_s_rhos).dot(
                    wt_invpsi_x)

            # M-step

            # Update Sigma_s and compute its trace
            sigma_s = (inv_sigma_s_rhos
                       + shared_response.dot(shared_response.T) / samples)
            trace_sigma_s = samples * np.trace(sigma_s)

            # Update each subject's mapping transform W_i and error variance
            # rho_i^2
            for subject in range(subjects):
                a_subject = x[subject].dot(shared_response.T)
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.001)
                u_subject, s_subject, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False)
                w[subject] = u_subject.dot(v_subject)
                rho2[subject] = trace_xtx[subject]
                rho2[subject] += -2 * np.sum(w[subject] * a_subject).sum()
                rho2[subject] += trace_sigma_s
                rho2[subject] /= samples * voxels[subject]

            if logger.isEnabledFor(logging.INFO):
                # Calculate and log the current log-likelihood for checking
                # convergence
                loglike = self._likelihood(
                    chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
                    trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
                    samples)
                logger.info('Objective function %f' % loglike)

        return sigma_s, w, mu, rho2, shared_response


class DetSRM(BaseEstimator, TransformerMixin):
    """Deterministic Shared Response Model (DetSRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    Note
    ----

        The number of voxels may be different between subjects. However, the
        number of samples must be the same across subjects.

        The Deterministic Shared Response Model is approximated using the
        Block Coordinate Descent (BCD) algorithm proposed in [Chen2015]_.

        This is a single node version.

        The run-time complexity is :math:`O(I (V T K + V K^2))` and the memory
        complexity is :math:`O(V T)` with I - the number of iterations, V - the
        sum of voxels from all subjects, T - the number of samples, K - the
        number of features (typically, :math:`V \\gg T \\gg K`), and N - the
        number of subjects.
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(self, X, y=None):
        """Compute the Deterministic Shared Response Model

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used
        """
        logger.info('Starting Deterministic SRM')

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects "
                             "({0:d}) to train the model.".format(len(X)))

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects"
                                 ".")

        # Run SRM
        self.w_, self.s_ = self._srm(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform data to the Shared Response subspace

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject.

        y : not used


        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _objective_function(self, data, w, s):
        """Calculate the objective function

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response

        Returns
        -------

        objective : float
            The objective function value.
        """
        subjects = len(data)
        objective = 0.0
        for m in range(subjects):
            objective += \
                np.linalg.norm(data[m] - w[m].dot(s), 'fro')**2

        return objective * 0.5 / data[0].shape[1]

    def _compute_shared_response(self, data, w):
        """ Compute the shared response S

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        Returns
        -------

        s : array, shape=[features, samples]
            The shared response for the subjects data with the mappings in w.
        """
        s = np.zeros((w[0].shape[1], data[0].shape[1]))
        for m in range(len(w)):
            s = s + w[m].T.dot(data[m])
        s /= len(w)

        return s

    def _srm(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.


        Returns
        -------

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response.
        """

        subjects = len(data)

        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject.
        w, _ = _init_w_transforms(data, self.features)
        shared_response = self._compute_shared_response(data, w)
        if logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(data, w, shared_response)
            logger.info('Objective function %f' % objective)

        # Main loop of the algorithm
        for iteration in range(self.n_iter):
            logger.info('Iteration %d' % (iteration + 1))

            # Update each subject's mapping transform W_i:
            for subject in range(subjects):
                a_subject = data[subject].dot(shared_response.T)
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.001)
                u_subject, _, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False)
                w[subject] = u_subject.dot(v_subject)

            # Update the shared response:
            shared_response = self._compute_shared_response(data, w)

            if logger.isEnabledFor(logging.INFO):
                # Calculate the current objective function value
                objective = self._objective_function(data, w, shared_response)
                logger.info('Objective function %f' % objective)

        return w, shared_response


class RSRM(BaseEstimator, TransformerMixin):
    """Robust Shared Response Model (RSRM)
    Given multi-subject data, factorize it as a shared response R among all
    subjects, an orthogonal transform W per subject, and an individual
    (outlying) sparse component S per subject:
    .. math:: X_i \\approx W_i R + S_i, \\forall i=1 \\dots N
    This unsupervised model allows to learn idiosyncratic information for
    subjects and simultaneously improve the shared response estimation.
    The model has similar properties to the Shared Response Model (SRM) with
    the addition of the individual components.
    The model is estimated solving the following optimization problem:
    .. math::
        \\min_{W_i, S_i, R}\\sum_i \\frac{1}{2}\\|X_i - W_i R - S_i\\|_F^2
    .. math:: + \\gamma\\|S_i\\|_1
    .. math:: s.t. \\qquad W_i^TW_i = I \\quad \\forall i=1 \\dots N
    The solution to this problem is obtained by applying a Block-Coordinate
    Descent procedure. More details can be found in [Turek2017]_.
    Parameters
    ----------
    n_iter : int, default: 10
        Number of iterations to run the algorithm.
    features : int, default: 50
        Number of features to compute.
    gamma : float, default: 1.0
        Regularization parameter for the sparseness of the individual
        components. Higher values yield sparser individual components.
    rand_seed : int, default: 0
        Seed for initializing the random number generator.
    Attributes
    ----------
    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.
    r_ : array, shape=[features, timepoints]
        The shared response.
    s_ : list of array, element i has shape=[voxels_i, timepoints]
        The individual components for each subject.
    random_state_: `RandomState`
        Random number generator initialized using rand_seed
    Note
    ----
        The number of voxels may be different between subjects. However, the
        number of timepoints for the alignment data must be the same across
        subjects.
        The Robust Shared Response Model is approximated using the
        Block-Coordinate Descent (BCD) algorithm proposed in [Turek2017]_.
        This is a single node version.
    """

    def __init__(self, n_iter=10, features=50, gamma=1.0, rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.gamma = gamma
        self.rand_seed = rand_seed

    def fit(self, X):
        """Compute the Robust Shared Response Model
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data of one subject.
        """
        logger.info('Starting RSRM')

        # Check that the regularizer value is positive
        if 0.0 >= self.gamma:
            raise ValueError("Gamma parameter should be positive.")

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects in the input "
                             "data to train the model.")

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough timepoints to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs for alignment
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of alignment timepoints "
                                 "between subjects.")

        # Create a new random state
        self.random_state_ = np.random.RandomState(self.rand_seed)

        # Run RSRM
        self.w_, self.r_, self.s_ = self._rsrm(X)

        return self

    def transform(self, X):
        """Use the model to transform new data to Shared Response space
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints_i]
            Each element in the list contains the fMRI data of one subject.
        Returns
        -------
        r : list of 2D arrays, element i has shape=[features_i, timepoints_i]
            Shared responses from input data (X)
        s : list of 2D arrays, element i has shape=[voxels_i, timepoints_i]
            Individual data obtained from fitting model to input data (X)
        """
        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        r = [None] * len(X)
        s = [None] * len(X)
        for subject in range(len(X)):
            if X[subject] is not None:
                r[subject], s[subject] = self._transform_new_data(X[subject],
                                                                  subject)

        return r, s

    def _transform_new_data(self, X, subject):
        """Transform new data for a subjects by projecting to the shared subspace and
        computing the individual information.
        Parameters
        ----------
        X : array, shape=[voxels, timepoints]
            The fMRI data of the subject.
        subject : int
            The subject id.
        Returns
        -------
        R : array, shape=[features, timepoints]
            Shared response from input data (X)
        S : array, shape=[voxels, timepoints]
            Individual data obtained from fitting model to input data (X)
        """
        S = np.zeros_like(X)
        R = None
        for i in range(self.n_iter):
            R = self.w_[subject].T.dot(X - S)
            S = self._shrink(X - self.w_[subject].dot(R), self.gamma)
        return R, S

    def transform_subject(self, X):
        """Transform a new subject using the existing model
        Parameters
        ----------
        X : 2D array, shape=[voxels, timepoints]
            The fMRI data of the new subject.
        Returns
        -------
        w : 2D array, shape=[voxels, features]
            Orthogonal mapping `W_{new}` for new subject
        s : 2D array, shape=[voxels, timepoints]
            Individual term `S_{new}` for new subject
        """
        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of TRs in the subject
        if X.shape[1] != self.r_.shape[1]:
            raise ValueError("The number of timepoints(TRs) does not match the"
                             "one in the model.")

        s = np.zeros_like(X)
        for i in range(self.n_iter):
            w = self._update_transform_subject(X, s, self.r_)
            s = self._shrink(X - w.dot(self.r_), self.gamma)

        return w, s

    def _rsrm(self, X):
        """Block-Coordinate Descent algorithm for fitting RSRM.
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data for alignment of
            one subject.
        Returns
        -------
        W : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.
        R : array, shape=[features, timepoints]
            The shared response.
        S : list of array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject.
        """
        subjs = len(X)
        voxels = [X[i].shape[0] for i in range(subjs)]
        TRs = X[0].shape[1]
        features = self.features

        # Initialization
        W = self._init_transforms(subjs, voxels, features, self.random_state_)
        S = self._init_individual(subjs, voxels, TRs)
        R = self._update_shared_response(X, S, W, features)

        if logger.isEnabledFor(logging.INFO):
            objective = self._objective_function(X, W, R, S, self.gamma)
            logger.info('Objective function %f' % objective)

        # Main loop
        for i in range(self.n_iter):
            W = self._update_transforms(X, S, R)
            S = self._update_individual(X, W, R, self.gamma)
            R = self._update_shared_response(X, S, W, features)
            # Print objective function every iteration
            if logger.isEnabledFor(logging.INFO):
                objective = self._objective_function(X, W, R, S, self.gamma)
                logger.info('Objective function %f' % objective)

        return W, R, S

    def _init_transforms(self, subjs, voxels, features, random_state):
        """Initialize the mappings (Wi) with random orthogonal matrices.
        Parameters
        ----------
        subjs : int
            The number of subjects.
        voxels : list of int
            A list with the number of voxels per subject.
        features : int
            The number of features in the model.
        random_state : `RandomState`
            A random state to draw the mappings.
        Returns
        -------
        W : list of array, element i has shape=[voxels_i, features]
            The initialized orthogonal transforms (mappings) :math:`W_i` for
            each subject.
        Note
        ----
            Not thread safe.
        """

        # Init the Random seed generator
        np.random.seed(self.rand_seed)

        # Draw a random W for each subject
        W = [random_state.random_sample((voxels[i], features))
             for i in range(subjs)]
        # Make it orthogonal it with QR decomposition
        for i in range(subjs):
            W[i], _ = np.linalg.qr(W[i])

        return W

    @staticmethod
    def _objective_function(X, W, R, S, gamma):
        """Evaluate the objective function.
        .. math:: \\sum_{i=1}^{N} 1/2 \\| X_i - W_i R - S_i \\|_F^2
        .. math:: + /\\gamma * \\|S_i\\|_1
        Parameters
        ----------
        X : list of array, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data for alignment of
            one subject.
        W : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.
        R : array, shape=[features, timepoints]
            The shared response.
        S : list of array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject.
        gamma : float, default: 1.0
            Regularization parameter for the sparseness of the individual
            components.
        Returns
        -------
        func : float
            The RSRM objective function evaluated on the parameters to this
            function.
        """
        subjs = len(X)
        func = .0
        for i in range(subjs):
            func += 0.5 * np.sum((X[i] - W[i].dot(R) - S[i])**2) \
                    + gamma * np.sum(np.abs(S[i]))
        return func

    @staticmethod
    def _update_individual(X, W, R, gamma):
        """Update the individual components `S_i`.
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data for alignment of
            one subject.
        W : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.
        R : array, shape=[features, timepoints]
            The shared response.
        gamma : float, default: 1.0
            Regularization parameter for the sparseness of the individual
            components.
        Returns
        -------
        S : list of array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject.
        """
        subjs = len(X)
        S = []
        for i in range(subjs):
            S.append(RSRM._shrink(X[i] - W[i].dot(R), gamma))
        return S

    @staticmethod
    def _init_individual(subjs, voxels, TRs):
        """Initializes the individual components `S_i` to empty (all zeros).
        Parameters
        ----------
        subjs : int
            The number of subjects.
        voxels : list of int
            A list with the number of voxels per subject.
        TRs : int
            The number of timepoints in the data.
        Returns
        -------
        S : list of 2D array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject initialized
            to zero.
        """
        return [np.zeros((voxels[i], TRs)) for i in range(subjs)]

    @staticmethod
    def _update_shared_response(X, S, W, features):
        """Update the shared response `R`.
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data for alignment of
            one subject.
        S : list of array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject.
        W : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.
        features : int
            The number of features in the model.
        Returns
        -------
        R : array, shape=[features, timepoints]
            The updated shared response.
        """
        subjs = len(X)
        TRs = X[0].shape[1]
        R = np.zeros((features, TRs))
        # Project the subject data with the individual component removed into
        # the shared subspace and average over all subjects.
        for i in range(subjs):
            R += W[i].T.dot(X[i]-S[i])
        R /= subjs
        return R

    @staticmethod
    def _update_transform_subject(Xi, Si, R):
        """Updates the mappings `W_i` for one subject.
        Parameters
        ----------
        Xi : array, shape=[voxels, timepoints]
            The fMRI data :math:`X_i` for aligning the subject.
        Si : array, shape=[voxels, timepoints]
            The individual component :math:`S_i` for the subject.
        R : array, shape=[features, timepoints]
            The shared response.
        Returns
        -------
        Wi : array, shape=[voxels, features]
            The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(R.T)
        A -= Si.dot(R.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    @staticmethod
    def _update_transforms(X, S, R):
        """Updates the mappings `W_i` for each subject.
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, timepoints]
            Each element in the list contains the fMRI data for alignment of
            one subject.ß
        S : list of array, element i has shape=[voxels_i, timepoints]
            The individual component :math:`S_i` for each subject.
        R : array, shape=[features, timepoints]
            The shared response.
        Returns
        -------
        W : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.
        """
        subjs = len(X)
        W = []
        for i in range(subjs):
            W.append(RSRM._update_transform_subject(X[i], S[i], R))
        return W

    @staticmethod
    def _shrink(v, gamma):
        """Soft-shrinkage of an array with parameter gamma.
        Parameters
        ----------
        v : array
            Array containing the values to be applied to the shrinkage operator
        gamma : float
            Shrinkage parameter.
        Returns
        -------
        v : array
            The same input array after the shrinkage operator was applied.
        """
        pos = v > gamma
        neg = v < -gamma
        v[pos] -= gamma
        v[neg] += gamma
        v[np.logical_and(~pos, ~neg)] = .0
        return v


