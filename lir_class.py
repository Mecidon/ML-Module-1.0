from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,Normalizer
from sklearn.linear_model import (LinearRegression,
                                  RidgeCV,
                                  LassoCV,
                                  ElasticNetCV)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
from Validation import Validation as Val

class LiR():
    """
    Class for performing linear regression and related tasks.
    """
    # Class variable used in export model function.
    today_date = date.today().strftime("%Y_%m_%d")

    def __init__(self,
                 X,
                 y,
                 degree:int=1,
                 test_size:float=0.3,
                 random_state:int=101,
                 scaling_first:bool=True,
                 standard_scaler:bool=False,
                 normal_scaler:bool=False,
                 ridgeCV:bool=False,
                 lassoCV:bool=False,
                 elasticNetCV:bool=False,
                 alphas:tuple=(0.01, 10, 0.1),
                 eps:float=0.001,
                 n_alphas:int=100,
                 max_iter:int=1000,
                 l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                 cv:int=10):
        """
        Initializer for Linear Regression model.

        Args:
            X: Features.
            y: Target label.
            degree: Degree of polynomial features
                (default is 1).
            test_size: Size of the test set 
                (default is 0.3).
            random_state: Random state for reproducibility
                (default is 101).
            standard_scaler: Whether to perform standard scaling
                (default is False).
            normal_scaler: Whether to perform normalization 
                (default is False).
            ridgeCV: Whether to use RidgeCV for regularization 
                (default is False).
            lassoCV: Whether to use LassoCV for regularization 
                (default is False).
            elasticNetCV: Whether to use ElasticNetCV for regularization 
                (default is False).
            alphas: Regularization strengths 
                (default is (0.01, 10, 0.1)).
            eps: Length of the path 
                (default is 0.001).
            n_alphas: Number of alphas along the regularization path 
                (default is 100).
            max_iter: Maximum number of iterations for optimization 
                (default is 1000).
            l1_ratio: Ratio of L1 penalty in the ElasticNet model 
                (default is [.1, .5, .7, .9, .95, .99, 1]).
        """
        self.X = X
        self.y = y
        self.degree = Val.validate_polynomial_degree(degree)
        self.test_size = Val.validate_test_size(test_size)
        self.random_state = random_state
        self.scaling_first = scaling_first
        self.standard_scaler = Val.validate_bool(standard_scaler)
        self.normal_scaler = Val.validate_bool(normal_scaler)
        self.ridgeCV = Val.validate_bool(ridgeCV)
        self.lassoCV = Val.validate_bool(lassoCV)
        self.elasticNetCV = Val.validate_bool(elasticNetCV)
        self.alphas = Val.validate_alphas(alphas)
        self.eps = Val.validate_not_bool(eps)
        self.n_alphas = Val.validate_not_bool(n_alphas)
        self.max_iter = Val.validate_not_bool(max_iter)
        self.l1_ratio = Val.validate_not_bool(l1_ratio)
        self.cv = cv

        # Train-test split
        self.X_train, self.X_test, \
        self.y_train, self.y_test = \
        train_test_split(self.X, self.y, \
        test_size=self.test_size, random_state=self.random_state)

        # Standard: If scaling is chosen before polynomial.
        if scaling_first:

            # Scaling
            if self.standard_scaler:
                self.s_scaler = StandardScaler()
                self.X_train = self.s_scaler.fit_transform(self.X_train)
                self.X_test = self.s_scaler.transform(self.X_test)

            if self.normal_scaler:
                self.n_scaler = Normalizer()
                self.X_train = self.n_scaler.fit_transform(self.X_train)
                self.X_test = self.n_scaler.transform(self.X_test)

            # Polynomial feature transformation
            self.polynomial_converter = PolynomialFeatures(degree=self.degree,
                                                           include_bias=False)
            self.X_train = self.polynomial_converter.fit_transform(self.X_train)
            self.X_test = self.polynomial_converter.fit_transform(self.X_test)

        # If polynomial is chosen before scaling.
        else:

            # Polynomial feature transformation
            self.polynomial_converter = PolynomialFeatures(degree=self.degree,
                                                           include_bias=False)
            self.X_train = self.polynomial_converter.fit_transform(self.X_train)
            self.X_test = self.polynomial_converter.fit_transform(self.X_test)

            if self.standard_scaler:
                self.s_scaler = StandardScaler()
                self.X_train = self.s_scaler.fit_transform(self.X_train)
                self.X_test = self.s_scaler.transform(self.X_test)

            if self.normal_scaler:
                self.n_scaler = Normalizer()
                self.X_train = self.n_scaler.fit_transform(self.X_train)
                self.X_test = self.n_scaler.transform(self.X_test)

        if self.elasticNetCV:
            # ElasticNetCV regression model
            self.model = ElasticNetCV(l1_ratio=self.l1_ratio,
                                      eps=self.eps,
                                      n_alphas=n_alphas,
                                      max_iter=self.max_iter,
                                      cv=self.cv)
        elif self.ridgeCV:
            # RidgeCV regression model
            self.model = RidgeCV(alphas=np.arange(self.alphas[0],
                                                  self.alphas[1],
                                                  self.alphas[2]),
                                 scoring="neg_mean_squared_error",
                                 cv=self.cv)
        elif self.lassoCV:
            # LassoCV regression model
            self.model = LassoCV(eps=self.eps,
                                 n_alphas=self.n_alphas,
                                 max_iter=self.max_iter,
                                 cv=self.cv)
        else:
            # Linear regression model
            self.model = LinearRegression(fit_intercept=True)

        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

        # Evaluation metrics
        self.MAE = mean_absolute_error(self.y_test, self.y_pred)
        self.MSE = mean_squared_error(self.y_test, self.y_pred)
        self.RMSE = np.sqrt(self.MSE)
        self.tolerance = self.RMSE / self.y.mean()
        self.model_r2_score = r2_score(self.y_test, self.y_pred)

        # Store scores in a dictionary
        self.scores = {"MAE": self.MAE, "MSE": self.MSE, "RMSE": self.RMSE,
                       "tolerance": self.tolerance, "r2_score": self.model_r2_score}

    def get_scores(self):
        """
        Print evaluation scores.
        """
        print("SCORES:")
        for key, val in self.scores.items():
            print(f"{key} : {val}")

    def get_poly_scores(self,degree_min:int=1, degree_max:int=8):
        """
        Print evaluation scores for polynomial features.
        
        Args:
            degree_min: Minimum degree of polynomial features (default is 1).
            degree_max: Maximum degree of polynomial features (default is 8).
        """
        # Loop through polynomial degrees
        for d in range(degree_min, degree_max):
            # Create polynomial features
            polynomial_converter = PolynomialFeatures(degree=d,
                                                      include_bias=False)
            poly_features = polynomial_converter.fit_transform(self.X)

            # Train-test split
            X_train, X_test, \
            y_train, y_test = train_test_split(poly_features,
                                               self.y,
                                               test_size=self.test_size,
                                               random_state=self.random_state)

            if self.standard_scaler == True:
                X_train = self.s_scaler.fit_transform(X_train)
                X_test = self.s_scaler.transform(X_test)

            if self.normal_scaler == True:
                X_train = self.n_scaler.fit_transform(X_train)
                X_test = self.n_scaler.transform(X_test)

            self.model.fit(X_train,y_train)
            y_pred = self.model.predict(X_test)
            print(f"{'-'*8}model with d_{d}{'-'*8}")
            #print(f"coef_ = {model.coef_}")
            print(f"MAE = {mean_absolute_error(y_true=y_test, y_pred=y_pred)}")
            print(f"RMSE = {mean_squared_error(y_true=y_test, y_pred=y_pred)**0.5}")
            print(f"tolerance = {mean_squared_error(y_true=y_test, y_pred=y_pred)**0.5/self.y.mean()}")
            print(f"r2_score = {r2_score(y_true=y_test, y_pred=y_pred)}")
            print(f'Number of total columns: {len(self.model.coef_)}')

    def residual_error_plot(self, title="Residual Error Plot",
                            xlabel="Actual Values", 
                            ylabel="Residual Error"):
        """
        Plot residual errors.

        Args:
            title: Title of the plot (default is "Residual Error Plot").
            xlabel: Label for the x-axis (default is "Actual Values").
            ylabel: Label for the y-axis (default is "Residual Error").
        """
        # Calculate residual errors
        residual_error = self.y_test - self.y_pred

        # Plotting with Seaborn regplot
        sns.scatterplot(x=self.y_test, y=residual_error)
        plt.axhline(y=0, color="red", linestyle="--")

        # Adding labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Show the plot
        plt.show()

    def rmse_plot(self,
                  degree_min:int=1,
                  degree_max:int=10,
                  test_size_loop:bool=False,
                  test_sizes:list=[0.1,0.2,0.3,0.4,0.5]):
        """
        Plot RMSE for different polynomial degrees.

        Args:
            degree_min: Minimum degree of polynomial features (default is 1).
            degree_max: Maximum degree of polynomial features (default is 10).
            test_size_loop: Whether to loop over test sizes (default is False).
            test_sizes: List of test sizes to loop over (default is [0.1,0.2,0.3,0.4,0.5]).
        """
        # If test_size_loop = True
        if test_size_loop:
            for test_size in test_sizes:
                train_rmse_errors = []
                test_rmse_errors = []
                for d in range(degree_min, degree_max):
                    # Polynomial converter
                    poly_features = PolynomialFeatures(degree=d,
                                                       include_bias=False).fit_transform(self.X)
                    # Train Test Split for chosen degree.
                    X_train, X_test, y_train, y_test = train_test_split(poly_features,
                                                                        self.y,
                                                                        test_size=test_size,
                                                                        random_state=self.random_state)
                    # Scaling if standard.
                    X_train = self.s_scaler.fit_transform(X_train) if self.standard_scaler else X_train
                    X_test = self.s_scaler.transform(X_test) if self.standard_scaler else X_test
                    # Scaling if normal.
                    X_train = self.n_scaler.fit_transform(X_train) if self.normal_scaler else X_train
                    X_test = self.n_scaler.transform(X_test) if self.normal_scaler else X_test
                    # Fitting model.
                    self.model.fit(X_train, y_train)
                    # Prediction variables.
                    train_pred, test_pred = self.model.predict(X_train), self.model.predict(X_test)
                    # Appending to list.
                    train_rmse_errors.append(np.sqrt(mean_squared_error(y_train, train_pred)))
                    test_rmse_errors.append(np.sqrt(mean_squared_error(y_test, test_pred)))
                    # Plotting.
                plt.plot(range(1, degree_max), train_rmse_errors, label=f"train, test_size={test_size}", marker="o")
                plt.plot(range(1, degree_max), test_rmse_errors, label=f"test, test_size={test_size}", marker="o")
                plt.title(f"RMSE/ Complexity/ test_size={test_size}")
                plt.xlabel("Polynomial Complexity")
                plt.ylabel("RMSE")
                plt.legend()
                plt.show()
        # If test_size_loop = False (Default)
        else:
            train_rmse_errors = []
            test_rmse_errors = []
            for d in range(degree_min, degree_max):
                poly_features = PolynomialFeatures(degree=d, include_bias=False).fit_transform(self.X)
                X_train, X_test, y_train, y_test = train_test_split(poly_features,
                                                                    self.y,
                                                                    test_size=self.test_size,
                                                                    random_state=self.random_state)
                X_train = self.s_scaler.fit_transform(X_train) if self.standard_scaler else X_train
                X_test = self.s_scaler.transform(X_test) if self.standard_scaler else X_test
                X_train = self.n_scaler.fit_transform(X_train) if self.normal_scaler else X_train
                X_test = self.n_scaler.transform(X_test) if self.normal_scaler else X_test
                self.model.fit(X_train, y_train)
                train_pred, test_pred = self.model.predict(X_train), self.model.predict(X_test)
                train_rmse_errors.append(np.sqrt(mean_squared_error(y_train, train_pred)))
                test_rmse_errors.append(np.sqrt(mean_squared_error(y_test, test_pred)))
            plt.plot(range(1, degree_max), train_rmse_errors, label="train", marker="o")
            plt.plot(range(1, degree_max), test_rmse_errors, label="test", marker="o")
            plt.title(f"RMSE/ Complexity/ test_size={self.test_size}")
            plt.xlabel("Polynomial Complexity")
            plt.ylabel("RMSE")
            plt.legend()
            plt.show()

    def plot_predictions(self, figsize:tuple = (15, 5)):
        """
        Plot predictions against actual values.

        Args:
            figsize: Figure size (default is (15, 5)).
        """
        # Creating substitute for self.X if using degree=1
        self.final_X = self.X

        if self.degree > 1:
            # final_X = poly_X
            self.final_X = self.polynomial_converter.fit_transform(self.final_X)

        if self.standard_scaler:
            # final_X = scaled_X
            self.final_X = self.s_scaler.fit_transform(self.final_X)

        if self.normal_scaler:
            # final_X = scaled_X
            self.final_X = self.n_scaler.fit_transform(self.final_X)

        # Train the final model
        self.model.fit(self.final_X, self.y)

        y_pred = self.model.predict(self.final_X)

        if self.model:
            fig, axes = plt.subplots(nrows=1,
                                    ncols=len(self.X.columns),
                                    figsize=figsize)

            # Display y-label on the leftmost picture
            axes[0].set_ylabel(f"{self.y.name} and {self.y.name} prediction")

            for i, col in enumerate(self.X.columns):
                axes[i].plot(self.X[col], self.y, "o", label="true value")
                axes[i].plot(self.X[col], y_pred, "o", color="red", label="prediction")
                axes[i].set_title(f"{col}")
                axes[i].legend()
            plt.tight_layout()
        else:
            raise ValueError("A valid model must be provided as an argument.")

    def export_model(self,
                     model_name="final_model",
                     converter_name="final_converter",
                     scaler_name="final_scaler"):
        """
        Export trained model, converter, and scaler.

        Args:
            model_name: Name of the model file (default is "final_model").
            converter_name: Name of the converter file (default is "final_converter").
            scaler_name: Name of the scaler file (default is "final_scaler").
        """
        if self.degree >1:
            converter_object = self.polynomial_converter
            converter_name = f"{converter_name}_{self.today_date}.joblib"
            dump(converter_object, converter_name)

        model_name = f"{model_name}_{self.today_date}.joblib"
        dump(self.model, model_name)

        if self.standard_scaler:
            standard_scaler_object = self.s_scaler
            s_scaler_name = f"{scaler_name}_standard_{self.today_date}.joblib"
            dump(standard_scaler_object, s_scaler_name)

        if self.normal_scaler:
            normal_scaler_object = self.n_scaler
            n_scaler_name = f"{scaler_name}_normal_{self.today_date}.joblib"
            dump(normal_scaler_object, n_scaler_name)

    @staticmethod
    def load_model(path:str):
        """
        Load a model from a selected file.

        Args:
            path: Path to the model file.

        Returns:
            Loaded model.
        """
        return load(f"{path}.joblib")
