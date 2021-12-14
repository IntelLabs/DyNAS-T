import pickle

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn import linear_model, svm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


class Predictor:

    def train(self, examples, labels):

        # Search for optimal regressor parameters
        self.searcher.fit(examples, labels)

        # Retrieve regressor trained with optimal parameters
        self.regressor = self.searcher.best_estimator_

    def predict(self, examples):
        '''
        Predicts the output values of the specified examples using the underlying regressor.

        Parameters
        ----------
            examples: Examples for which predictions will be made.

        Returns
        -------
            Predictions of the specified examples.
        '''

        # Return predictions
        return self.regressor.predict(examples)

    def get_parameters(self):
        '''
        Returns the optimal parameter values of the underlying regressor.

        Parameters
        ----------
            None

        Returns
        -------
            Optimal parameter values of the underlying regressor.
        '''

        # Retrieve optimal parameters
        return self.searcher.best_params_

    def get_metrics(self, examples, labels):
        '''
        Computes the performance metrics of the underlying regressor.

        Parameters
        ----------
            examples: Examples to use when computing performance metrics.

            labels: Labels to use when computing performance metrics.

        Returns
        -------
            Performance metrics of the underlying regressor. The metrics are
                Mean absolute percentage error (MAPE)
                Root mean squared error (RMSE)
                Kendall rank correlation coefficient (kendall)
                Spearman's rank correlation coefficient (spearman)
        '''

        # Compute predictions of specified examples
        predictions = self.predict(examples)

        # Compute performance metrics
        mape = 100.0 * mean_absolute_percentage_error(labels, predictions)
        rmse = mean_squared_error(labels, predictions, squared=False)
        kendall, _ = kendalltau(labels, predictions)
        spearman, _ = spearmanr(labels, predictions)

        return mape, rmse, kendall, spearman

    def load(self, filename):
        '''
        Loads the model of the underlying regressor and searcher.

        Parameters
        ----------
            filename: Name of the file from which to load the model.

        Returns
        -------
            None
        '''

        # Load searcher and regressor from specified file
        with open(filename, 'rb') as input_file:
            self.searcher = pickle.load(input_file)
            self.regressor = pickle.load(input_file)

    def save(self, filename):
        '''
        Saves the model of the underlying regressor and searcher.

        Parameters
        ----------
            filename: Name of the file to which to save the model.

        Returns
        -------
            None
        '''

        # Save searcher and regressor to specified file
        with open(filename, 'wb') as output_file:
            pickle.dump(self.searcher, output_file)
            pickle.dump(self.regressor, output_file)


class RidgePredictor(Predictor):

    def __init__(self, alphas, max_iterations, verbose):

        SEARCHER_VERBOSITY = 10

        # Create regressor
        self.regressor = linear_model.Ridge(max_iter=max_iterations)

        # Create parameter searcher
        search_parameters = {'alpha': alphas}
        self.searcher = GridSearchCV(estimator=self.regressor, param_grid=search_parameters, n_jobs=-1,
                                     scoring='neg_root_mean_squared_error', verbose=SEARCHER_VERBOSITY if (verbose) else 0)


class SVRPredictor(Predictor):

    def __init__(self, kernel_type, cost_factors, epsilons, max_iterations, verbose):

        SEARCHER_VERBOSITY = 10

        # Create regressor
        self.regressor = None
        if (kernel_type == 'linear'):
            self.regressor = svm.LinearSVR(max_iter=max_iterations)
        else:
            self.regressor = svm.SVR(kernel=kernel_type, max_iter=max_iterations)

        # Create parameter searcher
        search_parameters = {'C': cost_factors, 'epsilon': epsilons}
        self.searcher = GridSearchCV(estimator=self.regressor, param_grid=search_parameters, n_jobs=-1,
                                     scoring='neg_root_mean_squared_error', verbose=SEARCHER_VERBOSITY if (verbose) else 0)


class ResNet50AccuracyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)


class ResNet50LatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetAccuracyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetLatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetMACsPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.1, 0.6, 0.1)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)


class TransformerBleuPredictor(SVRPredictor):

    DEFAULT_COST_FACTORS = np.arange(1.0, 6.0, 1.0)
    DEFAULT_EPSILONS = np.arange(0.0, 0.55, 0.05)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, kernel_type='rbf', cost_factors=DEFAULT_COST_FACTORS, epsilons=DEFAULT_EPSILONS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(kernel_type, cost_factors, epsilons, max_iterations, verbose)


class TransformerLatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(5.0, 26.0, 1.0)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        super().__init__(alphas, max_iterations, verbose)
