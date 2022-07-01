from __future__ import annotations
from typing import Any, Iterable, Optional
import itertools as itt

import pandas as pd
import numpy as np

import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt


class Metrics:
    """
    Static class for regression metrics
    """

    @staticmethod
    def r_squared(estimated: list, obs: list) -> float:
        """
        Calculate r^2 value of an estimated set of data
        """
        rss, tss = 0, 0
        mean = np.average(obs)
        for i in range(len(obs)):
            rss += (obs[i] - estimated[i]) ** 2
            tss += (obs[i] - mean) ** 2
        return 1 - rss/tss

    @staticmethod
    def mse(estimated: list, obs: list) -> float:
        """
        Calculate the MSE of an estimated set of data
        """
        mse = 0
        for i in range(len(obs)):
            mse += (obs[i] - estimated[i])**2 
        return mse / len(obs)

    @staticmethod
    def mae(estimated: list, obs: list) -> float:
        """
        Calculate the MAE of an estimated set of data
        """
        mae = 0
        for i in range(len(obs)):
            mae += np.abs((estimated[i] - obs[i]))
        return mae / len(obs)


class ExponentialSmoothing:
    """
    Abstract superclass of all exponential smoothers.
    """

    def __init__(self, 
        dataset: np.array, 
        alpha: Optional[float] = 0,
        beta: Optional[float] = 0.001,
        gamma: Optional[float] = 0.25,
        phi: Optional[float] = 1,
        is_additive: Optional[bool] = False,
        season_length: Optional[int] = 12,
        initial_level: Optional[float] = None,
        initial_trend: Optional[float] = None,
        initial_season: tuple[float] = None,
        training_split: Optional[float] = 0.8,
        measure_of_error: Optional[str] = "MSE"
        ) -> None:
        bad_param_msg = (
            "Values for alpha, beta, gamma and phi must be between 0 and 1!"
        )

        bad_split_msg = (
            "Your training split is either too extreme, or your dataset is too small!"
        )

        bad_season_msg = (
            "Your season must be an integer value! e.g. 4, 12, 8..."
        )

        bad_initial_msg = (
            "Your input for an initial value (level/trend/season) is invalid!"
            "it might be that your intial season doesn't have 'm' values within it."
        )

        # Check for any values for model that are outside their 
        # possible ranges. If so, raise an error
        if not 0 <= alpha <= 1: raise ValueError(bad_param_msg)
        if not 0 <= beta <= 1: raise ValueError(bad_param_msg)
        if not 0 <= gamma <= 1: raise ValueError(bad_param_msg)
        if not 0 <= phi <= 1: raise ValueError(bad_param_msg)
        if not isinstance(season_length, int): raise ValueError(bad_season_msg)

        # Set all parameter values given that they fit
        # within their ranges.
        self._alpha = alpha
        self._beta = beta
        self._phi = phi
        self._gamma = gamma
        self._is_additive = is_additive
        self._dataset = np.array(dataset)
        
        # Get the index that we should split data by.
        # If it doesn't produce a good split, then send an error.
        self._split_index = int(len(dataset) * training_split)
        if not 0 < self._split_index < len(self._dataset): raise ValueError(bad_split_msg)
        
        # Set the trainset and testset
        self._trainset = np.array(self._dataset[:self._split_index])
        self._testset = np.array(self._dataset[self._split_index:])
        
        # Set the initial level if supplied. If not, we will take
        # the value at t = 0 as the initial value. There are estimation
        # methods for initial level, but not yet implemented.
        if initial_level: self._l0 = initial_level
        else: self._l0 = self._trainset[0]

        # Similiarly for initial trend
        if initial_trend: self._b0 = initial_trend
        else: self._b0 = self._dataset[1] - self._dataset[0]
        
        # Similiarly for intial season
        if initial_season and len(initial_season) != season_length: 
            raise ValueError(bad_initial_msg)
        elif initial_season: self._g0 = initial_season    
        else: 
            try: 
                self._g0 = self._trainset[:season_length]
            except IndexError:
                raise ValueError(bad_split_msg)
        self._m = season_length

        # Set our error measuring function. We can choose between
        # MSE, MAE and r^2. For HoltWinters regression, MSE and MAE 
        # are the best metrics for error.
        self._measure_of_error = measure_of_error
        if self._measure_of_error == "MSE": 
            self._err_func = Metrics.mse
        elif self._measure_of_error == "MAE": 
            self._err_func = Metrics.mae
        elif self._measure_of_error == "R_SQUARED": 
            self._err_func = Metrics.r_squared
        else: 
            self._err_func = Metrics.mae

        # Set up empty forecast value arrays. This will be populated
        # once the forecast function is called by the user.
        self._fcastvalues = np.zeros(len(self._dataset))
        self._is_optimal = False       
        self._all_data_used = False
        self._setup() # additional setup a specific smoothing tool may need to do

    def _setup(self,
        training_split: Optional[float] = 0.8,
        optimize: Optional[bool] = False,
        alpha: Optional[float] = 0.5,
        beta: Optional[float] = 0.1,
        gamma: Optional[float] = 0.1,
        phi: Optional[float] = None,
        seasonal_mode: Optional[str] = "add",
        season_length: Optional[int] = 4,
        initial_level: Optional[float] = None,
        initial_trend: Optional[float] = None,
        initial_season: Optional[Iterable[float]] = None,

        ) -> tuple:
        pass


    def forecast(self, 
        steps: int,
        use_all_data: Optional[bool] = False,
        optimize: Optional[bool] = False,
        use_library: Optional[bool] = False,
        decimal_accuracy: Optional[int] = 3
        ) -> list:
        """
        Forecasts the values for a 

        Parameters
        ----------
        steps : int
            number of steps forward we need to forecast (past the length
            of the dataset)
        optimize : Optional[bool], optional
            Whether to optimize parameters or not, by default False
        use_library : Optional[bool], optional
            Whether to use library or not, by default False
        decimal_accuracy : Optional[int]
            The decimal accuracy we want to optimize are parameters with

        Returns
        -------
        list
            List of y values that forecast the data
        """
        if steps < 0: raise ValueError("Number of steps must be 0 or positive!")

        # If we want to use all data to train, then
        # we set the trainset to be the whole dataset.
        # Might not be the best approach!!
        if use_all_data: 
            trainset = self._dataset
            testset = self._dataset
            split_idx = 0
            self._all_data_used = True
        else: 
            trainset = self._trainset
            testset = self._testset
            split_idx = self._split_index
            self._all_data_used = False
        if optimize: self._is_optimal = True
        if decimal_accuracy < 1: raise ValueError("Decimal accuracy must be atleast 1!")
        set_size = len(self._dataset) - len(trainset)

        # NOTE: if we are using library implementation,
        # we have limited manoueverability of the functions.
        # Might be better to use library functions outside of 
        # this set.
        if use_library:
            return [0]
            model = self._lmodel_func(
                trainset,
                self._alpha,
                self._beta,
                self._gamma,
                self._phi,
                optimize
            )
            forecast = model.predict(
                start = 0,
                end = len(self._dataset) + steps
            )
            self._fcastvalues = forecast
            self._alpha = model.params["smoothing_level"]
            self._beta = model.params["smoothing_trend"]
            self._gamma = model.params["smoothing_seasonal"]
            self._phi = model.params["damping_trend"]
            return forecast
        
        # Here we are optimizing for all the models parameters
        # The parameters that we may optimize are:
        # Alpha, Beta, Phi, Gamma. Depends on which function
        if optimize:
            opti_set = self._get_optiset(decimal_accuracy)
            best_params = None
            min_error = np.inf

            # Iterate over all combinations of all parameters.
            # Depending on the type of exponential smoothing,
            # We will have less / more params to optimize
            for params in opti_set:
                current_fc = self._predict(trainset, params, set_size)
                test_error = self._err_func(testset, current_fc[split_idx:])
                if test_error < min_error:
                    best_params = params
                    min_error = test_error
            
            self.set_params(best_params)
        
        use_params = self.get_params()
        forecastvalues = self._predict(trainset, use_params, set_size + steps)
        self._fcastvalues = forecastvalues.copy()
        return forecastvalues

    def __get_errors(self):
        """
        Gets training, testing and total errors of model (in that order)
        """
        forecast = self._fcastvalues.copy()[:len(self._dataset)]
        train_err = self._err_func(forecast[:self._split_index], self._trainset)
        test_err = self._err_func(forecast[self._split_index:], self._testset)
        total_err = self._err_func(forecast, self._dataset)
        return {"train": train_err, "test": test_err, "total": total_err}

    def get_summary(self) -> str:
        """
        Returns a summary string about the model. 
        """
        errors = self.__get_errors()
        ret = f"---- {self.__class__.__name__} Model ----\n"
        ret += f"Measure of error: {self._measure_of_error}\n"
        if not self._all_data_used:
            ret += f"Training error: {errors['train']}\n"
            ret += f"Testing error: {errors['test']}\n"
        ret += f"Total error: {errors['total']}\n"
        ret += f"Optimized?: {self._is_optimal}\n"
        ret += f"Alpha: {self._alpha}\n"
        return ret     

    def _predict(self, trainset, params, steps):
        """ 
        To be implemented by subclasses. Regressor function.
        """
        raise NotImplementedError

    def _get_optiset(self, accuracy: int) -> np.array:
        """
        To be implemented be implemented by subclass. Gets
        the set of all param configs such that the optimal
        can be found
        """
        raise NotImplementedError

    def set_params(self, 
        params: Optional[np.array] = None):
        """
        Set the parameter values for the function
        """
        raise NotImplementedError

    def get_params(self) -> Any:
        """
        Get the current parameter values for the model
        """
        raise NotImplementedError


class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple exponential smoothing forecast method (Flat forecast).
    """
    REGRESSOR = smt.SimpleExpSmoothing

    def _setup(self):
        self._lmodel_func = (lambda dataset, l0, alpha, beta, gamma, phi, optimize : 
            SimpleExpSmoothing.REGRESSOR(
                dataset,
                initialization_method = "known",
                initial_level = l0).fit(
                smoothing_level = alpha,
                optimized = optimize
            )
        )

    def _predict(self, 
        trainset: list, 
        alpha: float,
        steps: int
        ) -> list:
        """
        Returns list of levels up to the size of the set given
        """
        if steps < 0: raise ValueError("Number of steps must be 0 or positive!")
        train_len = len(trainset)
        levels = np.zeros(train_len + 1 + steps, dtype = float)
        levels[0] = self._l0
        
        for i in range(train_len):
            levels[i + 1] = alpha * trainset[i] + (1 - alpha) * levels[i]

        forecast_val = levels[train_len]
        for i in range(train_len, steps + train_len):
            levels[i] = forecast_val

        return levels[:-1]

    def _get_optiset(self, accuracy: int) -> np.array:
        """
        Get an array of all possible alpha configurations
        depending on the accuracy given.
        """
        alphas = np.zeros(10**accuracy, dtype = float)
        
        c_alpha = 1 / (10 ** accuracy)
        inc = c_alpha
        for i in range(10 ** accuracy - 1):
            alphas[i] = round(c_alpha, accuracy)
            c_alpha += inc

        return alphas

    def set_params(self, alpha: np.array) -> None:
        """ For a SingleExpSmooth model, set alpha value """
        self._alpha = alpha

    def get_params(self) -> float:
        """ Get the current parameter values for the model """
        return self._alpha


class DoubleExpSmoothing(ExponentialSmoothing):
    """
    Double Exponentional Smoothing forecast (linear trend forecast). 
    Has the option of damping.
    """
    REGRESSOR = smt.Holt

    def _setup(self):
        self._lmodel_func = (lambda dataset, alpha, beta, gamma, phi, optimize : 
            SimpleExpSmoothing.REGRESSOR(
                dataset,
                initialization_method = "heuristic"
                ).fit(
                smoothing_level = alpha,
                smoothing_trend = beta,
                optimized = optimize
            )
        )

    def _predict(self, 
        trainset: list, 
        params: float,
        steps: int
        ) -> list:
        """
        Returns forecasted values from t = 0 to t = t + steps
        """
        if steps < 0: raise ValueError("Number of steps must be 0 or positive!")
        train_len = len(trainset)

        alpha = params[0]
        beta = params[1]

        levels = np.zeros(train_len + 1, dtype = float)
        levels[0] = self._l0
        trends = np.zeros(train_len + 1, dtype = float)
        trends[0] = self._b0

        forecast = np.zeros(len(self._dataset) + steps, dtype = float)
        
        for i in range(train_len):
            levels[i + 1] = alpha * trainset[i] + (1 - alpha) * (levels[i] + trends[i])
            trends[i + 1] = beta * (levels[i + 1] - levels[i]) + (1 - beta) * trends[i]
            forecast[i] = levels[i] + trends[i]
 
        last_level = levels[train_len]
        last_trend = trends[train_len]
        h = 1
        for i in range(train_len, steps + train_len):
            forecast[i] = last_level + h * last_trend
            h += 1

        return forecast

    def _get_optiset(self, accuracy: int) -> np.array:
        """
        Get an array of all possible params configurations
        depending on the accuracy given.
        """
        num_rows = 10**accuracy
        params = np.zeros(((num_rows - 1) ** 2, 2), dtype = float)
        
        c_alpha = 1 / num_rows
        c_beta = 1 / num_rows
        inc = c_alpha
        for i in range(num_rows - 1):
            for j in range(num_rows - 1):
                params[i * (num_rows - 1) + j][0] = c_alpha
                params[i * (num_rows - 1) + j][1] = c_beta
                c_beta = round(c_beta + inc, 2)
            c_alpha = round(c_alpha + inc, 2)
            c_beta = 1 / num_rows

        return params

    def set_params(self, params: np.array) -> None:
        """ For a SingleExpSmooth model, set alpha value """
        self._alpha = params[0]
        self._beta = params[1]

    def get_params(self) -> float:
        """ Get the current parameter values for the model """
        return (self._alpha,)

    def get_summary(self) -> str:
        ret = super().get_summary()
        ret += f"Beta: {self._beta}\n"
        # TODO: Fix damping
        if True:
            ret += f"Phi: {self._phi}\n"
        return ret


class HoltWinters(ExponentialSmoothing):
    """
    Holt Winters exponential smoothing with seasonality.
    """
    REGRESSOR = smt.ExponentialSmoothing

    def _setup(self):
        if self._is_additive: mode = "add"
        else: mode = "mul"
        self._lmodel_func = (lambda dataset, alpha, beta, gamma, phi, optimize : 
            HoltWinters.REGRESSOR(
                dataset,
                initialization_method = "heuristic",
                seasonal = mode,
                seasonal_periods = self._m
                ).fit(
                smoothing_level = alpha,
                smoothing_trend = beta,
                smoothing_seasonal = gamma,
                optimized = optimize
            )
        )

    def _predict(self, 
        trainset: list, 
        params: float,
        steps: int
        ) -> list:
        """
        Returns forecasted values from t = 0 to t = t + steps
        """

        train_len = len(trainset)
        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        m = int(params[3])

        levels = np.zeros(train_len + 1, dtype = float)
        levels[0] = self._l0
        trends = np.zeros(train_len + 1, dtype = float)
        trends[0] = self._b0
        seasons = np.zeros(train_len + m, dtype = float)
        seasons = np.insert(seasons, 0, self._g0)
        forecast = np.zeros(len(self._dataset) + steps, dtype = float)
        
        for i in range(train_len):
            if self._is_additive:
                levels[i + 1] = alpha * (trainset[i] - seasons[i]) + (1 - alpha) * (levels[i] + trends[i])
                trends[i + 1] = beta * (levels[i + 1] - levels[i]) + (1 - beta) * trends[i]
                seasons[i + m + 1] = gamma * (trainset[i] - levels[i] - trends[i]) + (1 - gamma) * seasons[i + m]
                forecast[i] = levels[i] + trends[i] + seasons[i]
            else:
                levels[i + 1] = alpha * (trainset[i] / seasons[i]) + (1 - alpha) * (levels[i] + trends[i])
                trends[i + 1] = beta * (levels[i + 1] - levels[i]) + (1 - beta) * (trends[i])
                seasons[i + 1] = gamma * (trainset[i] / (levels[i] + trends[i])) + (1 - gamma) * seasons[i]
                forecast[i] = (levels[i] + trends[i]) / seasons[i]

        last_level = levels[-1]
        last_trend = trends[-1]
        last_season = seasons[-1 * m:]
        h = 1
        
        for i in range(train_len, steps + train_len):
            if self._is_additive:
                forecast[i] = last_level + h * last_trend + last_season[h % m]
            else:
                forecast[i] = (last_level + h * last_trend) * last_season[h % m]

        return forecast

    def _get_optiset(self, accuracy: int) -> np.array:
        """
        Get an array of all possible params configurations
        depending on the accuracy given.
        """
        num_rows = 10**accuracy        
        vals = np.zeros(num_rows - 1, dtype = float)
        a = 1 / num_rows
        inc = 1 / num_rows
        for i in range(num_rows - 1):
            vals[i] = a
            a = round(a + inc, accuracy)
        ms = [x for x in range(1, self._m + 1)]

        ret = list(itt.product(vals, vals, vals, ms))
        return np.array(ret, dtype = float)

    def set_params(self, params: np.array) -> None:
        """ set params """
        self._alpha = params[0]
        self._beta = params[1]
        self._alpha = params[2]
        self._m = params[3]

    def get_params(self) -> float:
        """ Get the current parameter values for the model """
        return (self._alpha, self._beta, self._gamma, self._m)

    def get_summary(self) -> str:
        ret = super().get_summary()
        ret += f"Beta: {self._beta}\n"
        ret += f"Gamma: {self._gamma}\n"
        ret += f"Season length: {self._m}\n"
        return ret


if __name__ == "__main__":
    df = pd.read_csv("airline.csv")
    data = df["sales"]
    #actual = df["predicted"]

    ses = SimpleExpSmoothing(
        data, 
        training_split = 0.8
    )
    forecast = ses.forecast(
        10,
        optimize = True,
        use_library = False,
        use_all_data = False
    )
    forecast = np.array([round(x, 2) for x in forecast], dtype = float)
    ses2 = SimpleExpSmoothing(data)
    forecast2 = ses2.forecast(
        10,
        optimize = True,
        use_library = True,
        use_all_data = True
    )
    print(ses.get_summary())
    print(ses2.get_summary())
    
    #des = DoubleExpSmoothing(
    #    data, 
    #    alpha = 0.8321, 
    #    beta = 0.0001,
    #)
    #forecast = des.forecast(
    #    5, 
    #    optimize = True,
    #    use_library = False,
    #    use_all_data = False,
    #    decimal_accuracy = 1
    #)
    #des2 = DoubleExpSmoothing(
    #    data, 
    #    alpha = 0.8321, 
    #    beta = 0.0001,
    #)
    #forecast2 = des2.forecast(
    #    5, 
    #    optimize = True,
    #    use_library = True,
    #    use_all_data = False,
    #    decimal_accuracy = 3
    #)
    
    #print(des.get_summary())
    #print(des2.get_summary())


    #tes = HoltWinters(
    #    data, 
    #    training_split = 0.7,
    #    alpha = 0.8321, 
    #    beta = 0.0001,
    #    gamma = 0.5,
    #    season_length = 4,
    #    is_additive = True  
    #)
    #forecast = tes.forecast(
    #    8, 
    #    optimize = True,
    #    use_library = False,
    #    use_all_data = True,
    #    decimal_accuracy = 1
    #)
    #tes2 = HoltWinters(
    #    data, 
    #    alpha = 0.8321, 
    #    beta = 0.0001,
    #    gamma = 0.5,
    #    season_length = 4,
    #    is_additive = True  
    #)
    #forecast2 = tes2.forecast(
    #    8, 
    #    optimize = False,
    #    use_library = True,
    #    use_all_data = True,
    #    decimal_accuracy = 1
    #)
    
    #print(tes.get_summary())
    #print(tes2.get_summary())

    #print(forecast)
    #print(forecast2)

    plt.subplot(1, 2, 1)
    plt.legend(["forecast", "observed"])
    plt.xlabel("Time (Months)")
    plt.ylabel("Airline Passengers")
    plt.plot([x for x in range(len(forecast))], forecast, linestyle="dashed")
    plt.plot([x for x in range(len(data))], data)
    plt.axvline(int(len(data) * 0.8))
    plt.title("Without library")

    plt.subplot(1, 2, 2)
    plt.legend(["forecast", "observed"])
    plt.xlabel("Time (Months)")
    plt.ylabel("Airline Passengers")
    plt.plot([x for x in range(len(forecast2))], forecast2, linestyle="dashed")
    plt.plot([x for x in range(len(data))], data)
    plt.axvline(int(len(data) * 0.8))
    plt.title("With Library")
    plt.show()