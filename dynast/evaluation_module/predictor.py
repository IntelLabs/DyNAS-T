import numpy as np
import pickle
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

    def predict_single(self, example):
        
        return self.regressor.predict(example.reshape(1,-1))[0]  

    def load(self, filename):

        # Load searcher and regressor from specified file
        with open(filename, 'rb') as input_file:
            self.searcher = pickle.load(input_file)
            self.regressor = pickle.load(input_file)

    def save(self, filename):

        # Save searcher and regressor to specified file
        with open(filename, 'wb') as output_file:
            pickle.dump(self.searcher, output_file)
            pickle.dump(self.regressor, output_file)


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
        self.searcher = GridSearchCV(estimator = self.regressor, param_grid = search_parameters, n_jobs = -1,
            scoring = 'neg_root_mean_squared_error', verbose = SEARCHER_VERBOSITY if (verbose) else 0)


class ResNet50AccuracyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class ResNet50LatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetAccuracyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetLatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.5, 2.5, 0.5)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class MobileNetMACsPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(0.1, 0.6, 0.1)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)


class TransformerBleuPredictor(SVRPredictor):

    DEFAULT_COST_FACTORS = np.arange(1.0, 6.0, 1.0)
    DEFAULT_EPSILONS = np.arange(0.0, 0.55, 0.05)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, kernel_type = 'rbf', cost_factors = DEFAULT_COST_FACTORS, epsilons = DEFAULT_EPSILONS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(kernel_type, cost_factors, epsilons, max_iterations, verbose)


class TransformerLatencyPredictor(RidgePredictor):

    DEFAULT_ALPHAS = np.arange(5.0, 26.0, 1.0)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas = DEFAULT_ALPHAS, max_iterations = DEFAULT_MAX_ITERATIONS, verbose = False):

        super().__init__(alphas, max_iterations, verbose)
