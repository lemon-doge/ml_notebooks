import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from distributions import BaseDistribution, NumericalDistribution, CategoricalDistribution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF


class BaseOptimizer(BaseEstimator, ABC):
    '''
    A base class for all hyperparameter optimizers
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, n_jobs: Optional[int] = None,
                 verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - estimator: sklearn model instance
          - param_distributions: a dictionary of parameter distributions,
            e.g. param_distributions['num_epochs'] = IntUniformDistribution(100, 200)
          - scoring: sklearn scoring object, see
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            if left None estimator must have 'score' attribute
          - cv: number of folds to cross-validate
          - num_runs: number of iterations to fit hyperparameters
          - num_dry_runs: number of dry runs (i.e. random strategy steps) to gather initial statistics
          - num_samples_per_run: number of hyperparameters set to sample each iteration
          - n_jobs: number of parallel processes to fit algorithms
          - verbose: whether to print debugging information (you can configure debug as you wish)
          - random_state: RNG seed to control reproducibility
        '''
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.num_runs = num_runs
        self.num_samples_per_run = num_samples_per_run
        self.num_dry_runs = num_dry_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        # чтобы не подсвечивалось убого
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.array([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])
        # seed np random
        np.random.seed(self.random_state)

    def reset(self):
        '''
        Reset fields used for fitting
        '''
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.array([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])

    def sample_params(self) -> Dict[str, np.array]:
        '''
        Sample self.num_samples_per_run set of hyperparameters
        Returns:
          - sampled_params: dict of arrays of parameter samples,
            e.g. sampled_params['num_epochs'] = np.array([178, 112, 155])
        '''
        sampled_params = {}

        for param_name, param_distr in self.param_distributions.items():
            sampled_params[param_name] = param_distr.sample(self.num_samples_per_run)

        return sampled_params

    @abstractmethod
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        msg = f'method \"select_params\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def cross_validate(self, X: np.array, y: Optional[np.array],
                       params: Dict[str, Any]):
        '''
        Calculate cross-validation score for a set of params
        Consider using estimator.set_params() and sklearn.model_selection.cross_validate()
        Also use self.splitter as a cv parameter in cross_validate
        Params:
          - X: object features
          - y: object labels
          - params: a set of params to score
        Returns:
          - score: mean cross-validation score
        '''
        # Your code here (⊃｡•́‿•̀｡)⊃

        # fixing seed
        if 'random_state' in self.estimator.get_params().keys():
            self.estimator = self.estimator.set_params(**{'random_state': self.random_state})

        return np.mean(cross_validate(estimator=self.estimator.set_params(**params),
                                      X=X,
                                      y=y,
                                      cv=self.splitter,
                                      n_jobs=self.n_jobs)['test_score'])

    def fit(self, X_train: np.array, y_train: Optional[np.array] = None) -> BaseEstimator:
        '''
        Find the best set of hyperparameters with a specific search strategy
        using cross-validation and fit self.best_estimator on whole training set
        Params:
          - X_train: array of train features of shape (num_samples, num_features)
          - y_train: array of train labels of shape (num_samples, )
            if left None task is unsupervised
        Returns:
          - self: (sklearn standard convention)
        '''
        self.reset()

        # seed np random
        np.random.seed(self.random_state)

        if y_train is not None and np.issubdtype(y_train.dtype, np.integer):
            self.splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)

        # Your code here (⊃｡•́‿•̀｡)⊃

        # finding best parameters
        for i in range(self.num_runs):
            sampled_params = self.sample_params()

            selected_params = self.select_params(self.params_history, self.scores_history, sampled_params)

            for name in selected_params.keys():
                self.params_history[name] = np.append(self.params_history[name], selected_params[name])

            score = self.cross_validate(X_train, y_train, selected_params)
            self.scores_history = np.append(self.scores_history, score)

            if self.best_score is None or self.best_score < score:  # <= проверил
                self.best_score = score
                self.best_params = selected_params

        # setting up best estimator
        self.best_estimator = self.estimator.set_params(**self.best_params)
        # fixing seed
        if 'random_state' in self.best_estimator.get_params().keys():
            self.best_estimator = self.best_estimator.set_params(**{'random_state': self.random_state})
        # fitting best estimator
        self.best_estimator = self.best_estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.array) -> np.array:
        '''
        Generate a prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test predictions of shape (num_samples, )
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        # your code here ┐(シ)┌
        y_pred = self.best_estimator.predict(X_test)
        return y_pred

    def predict_proba(self, X_test: np.array) -> np.array:
        '''
        Generate a probability prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test probabilities of shape (num_samples, num_classes)
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        if not hasattr(self.best_estimator, 'predict_proba'):
            raise ValueError('Estimator does not support predict_proba')

        # your code here ┐(シ)┌
        y_pred = self.best_estimator.predict_proba(X_test)
        return y_pred


class RandomSearchOptimizer(BaseOptimizer):
    '''
    An optimizer implementing random search strategy
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        super().__init__(
            estimator, param_distributions, scoring, cv=cv,
            num_runs=num_runs, num_dry_runs=0, num_samples_per_run=1,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        j = np.random.randint(0, self.num_samples_per_run)
        for name in sampled_params.keys():
            new_params[name] = sampled_params[name][j]
        return new_params


class GPOptimizer(BaseOptimizer):
    '''
    An optimizer implementing gaussian process strategy
    '''

    @staticmethod
    def calculate_expected_improvement(y_star: float, mu: np.array,
                                       sigma: np.array) -> np.array:
        '''
        Calculate EI values for passed parameters of normal distribution
        hint: consider using scipy.stats.norm
        Params:
          - y_star: optimal (maximal) score value
          - mu: array of mean values of normal distribution of size (num_samples_per_run, )
          - sigma: array of std values of normal distribution of size (num_samples_per_run, )
        Retuns:
          - ei: array of EI values of size (num_samples_per_run, )
        '''
        ei = np.zeros_like(mu)
        # Your code here (⊃｡•́‿•̀｡)⊃
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/
        with np.errstate(divide='warn'):
            imp = mu - y_star
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def calculate_best_categorical(self, param_name: str, y_star: float, params_history: np.array,
                                   scores_history: np.array):
        '''
        returns best value for given categorical parameter
        '''
        if not issubclass(self.param_distributions[param_name].__class__, CategoricalDistribution):
            raise ValueError('non-categorical parameter given')

        categories = self.param_distributions[param_name].categories

        mu = np.zeros(len(categories))
        sigma = np.zeros(len(categories))

        for i in range(len(categories)):
            occurrences = (params_history[param_name] == categories[i])
            occurrences_sum = np.sum(occurrences)
            if occurrences_sum == 0:
                mu[i] = y_star
            else:
                weighted_sum = np.sum(occurrences * scores_history)
                mu[i] = weighted_sum / occurrences_sum

            sigma[i] = (1 + np.sum(occurrences * np.square(scores_history - mu[i]))) / (1 + occurrences_sum)

        ei = self.calculate_expected_improvement(y_star, mu, sigma)

        return categories[np.argmax(ei)]

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃

        # num_dry_runs включены в num_runs
        if len(scores_history) < self.num_dry_runs:
            j = np.random.randint(0, self.num_samples_per_run)
            for name in sampled_params.keys():
                new_params[name] = sampled_params[name][j]
            return new_params


        numerical_param_names = [name for name in params_history.keys() if
                                 issubclass(self.param_distributions[name].__class__, NumericalDistribution)]
        categorical_param_names = [name for name in params_history.keys() if
                                   issubclass(self.param_distributions[name].__class__, CategoricalDistribution)]

        y = scores_history
        y_star = np.max(scores_history)

        X_numerical = np.stack([self.param_distributions[name].scale(params_history[name]) for name in
                                numerical_param_names]).T  # векторы параметров по строчкам

        gpr = GaussianProcessRegressor(kernel=ConstantKernel() + WhiteKernel() + RBF(),
                                       random_state=self.random_state).fit(X_numerical, y)

        new_X_numerical = np.stack([self.param_distributions[name].scale(sampled_params[name]) for name in
                                    numerical_param_names]).T  # векторы параметров по строчкам

        mu_numerical, sigma_numerical = gpr.predict(new_X_numerical, return_std=True)

        ei = self.calculate_expected_improvement(y_star, mu_numerical, sigma_numerical)

        argmax_ei = np.argmax(ei)

        for name in numerical_param_names:
            new_params[name] = sampled_params[name][argmax_ei]

        for name in categorical_param_names:
            new_params[name] = self.calculate_best_categorical(name, y_star, params_history, scores_history)

        return new_params


class TPEOptimizer(BaseOptimizer):
    '''
    An optimizer implementing tree-structured Parzen estimator strategy
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, gamma: float = 0.75,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - gamma: scores quantile used for history splitting
        '''
        super().__init__(
            estimator, param_distributions, scoring, cv=cv, num_runs=num_runs,
            num_dry_runs=num_dry_runs, num_samples_per_run=num_samples_per_run,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
        self.gamma = gamma

    @staticmethod
    def estimate_log_density(scaled_params_history: np.array,
                             scaled_sampled_params: np.array, bandwidth: float):
        '''
        Estimate log density of sampled numerical hyperparameters based on
        numerical hyperparameters history subset
        Params:
          - scaled_params_history: array of scaled numerical hyperparameters history subset
            of size (subset_size, num_numerical_params)
          - scaled_sampled_params: array of scaled sampled numerical hyperparameters
            of size (num_samples_per_run, num_numerical_params)
          - bandwidth: bandwidth for KDE
        Returns:
          - log_density: array of estimated log probabilities of size (num_samples_per_run, )
        '''
        log_density = np.zeros(self.num_samples_per_run)
        # Your code here (⊃｡•́‿•̀｡)⊃
        return log_density

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        return new_params
