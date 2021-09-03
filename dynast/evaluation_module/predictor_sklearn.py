import numpy as np
from sklearn import svm, linear_model
from sklearn.model_selection import GridSearchCV


class Predictor:

    def train(self, examples, labels):

        # Search for optimal regressor parameters
        self.searcher.fit(examples, labels)

        # Retrieve regressor trained with optimal parameters
        self.regressor = self.searcher.best_estimator_


    def predict(self, examples):

        return self.regressor.predict(examples)


class RidgePredictor(Predictor):

    def __init__(self, alphas, max_iterations, verbose):

        SEARCHER_VERBOSITY = 10

        # Create regressor
        self.regressor = linear_model.Ridge(max_iter = max_iterations)

        # Create parameter searcher
        search_parameters = {'alpha': alphas}
        self.searcher = GridSearchCV(estimator = self.regressor, param_grid = search_parameters, n_jobs = -1,
            scoring = 'neg_root_mean_squared_error', verbose = SEARCHER_VERBOSITY if (verbose) else 0)


class SVRPredictor(Predictor):

    def __init__(self, kernel_type, cost_factors, epsilons, max_iterations, verbose):

        SEARCHER_VERBOSITY = 10

        # Create regressor
        self.regressor = None
        if (kernel_type == 'linear'):
            self.regressor = svm.LinearSVR(max_iter = max_iterations)
        else:
            self.regressor = svm.SVR(kernel = kernel_type, max_iter = max_iterations)

        # Create parameter searcher
        search_parameters = {'C': cost_factors, 'epsilon': epsilons}
        self.searcher = GridSearchCV(estimator = self.regressor, param_grid = search_parametepythonrs, n_jobs = -1,
            scoring = 'neg_root_mean_squared_error', verbose = SEARCHER_VERBOSITY if (verbose) else 0)


class MobileNetLatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 5.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetAccuracyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(1.0, 16.0, 1.0)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class TransformerBleuPredictor(SVRPredictor):

    DEFAULT_COST_FACTORS = np.arange(1.0, 6.0, 1.0)
    DEFAULT_EPSILONS = np.arange(0.0, 0.95, 0.05)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, kernel_type = 'rbf', cost_factors = DEFAULT_COST_FACTORS, epsilons = DEFAULT_EPSILONS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(kernel_type, cost_factors, epsilons, max_iterations, verbose)
