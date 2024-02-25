from datetime import date
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, 
                             accuracy_score,
                             confusion_matrix, 
                             roc_curve, auc,
                             ConfusionMatrixDisplay, 
                             PrecisionRecallDisplay, 
                             RocCurveDisplay)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from Validation import Validation as Val

class LoR():
    """
    Class for performing logistic regression and related tasks.
    """
    # Class variable used in export model function.
    today_date = date.today().strftime("%Y_%m_%d")

    def __init__(self,
                 X,
                 y,
                 degree: int = 1,
                 test_size: float = 0.3,
                 random_state: int = 101,
                 scaling_first: bool = True,
                 standard_scaler: bool = False,
                 normal_scaler: bool = False,
                 logistic: bool = False,
                 logistic_cv: bool = False,
                 Cs=10,
                 C=np.logspace(0, 1, 10),
                 cv=10,
                 penalty_cv="elasticnet",
                 penalty=['l1', 'l2', 'elasticnet'],
                 scoring="accuracy",
                 solver_cv="saga",
                 solver=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                 max_iter_cv=1000,
                 max_iter=[1000],
                 multi_class_cv="auto",
                 multi_class=['auto', 'ovr', 'multinomial'],
                 class_weight_cv=None,
                 class_weight=[None],
                 l1_ratios=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                 l1_ratio=[None]
                 ):
        """
        Initialize Logistic Regression model.

        Args:
            X: Features.
            y: Target label.
            degree: Degree of polynomial features
                (default is 1).
            test_size: Size of the test set
                (default is 0.3).
            random_state: Random state for reproducibility 
                (default is 101).
            scaling_first: Whether to perform scaling first 
                (default is True).
            standard_scaler: Whether to perform standard scaling 
                (default is False).
            normal_scaler: Whether to perform normalization 
                (default is False).
            logistic: Whether to use Logistic Regression 
                (default is False).
            logistic_cv: Whether to use Logistic Regression with Cross Validation 
                (default is False).
            Cs: List of floats or int of values to be tested as hyperparameters for logistic regression 
                (default is 10).
            C: Regularization strength (default is np.logspace
                (0, 1, 10)).
            cv: Cross-validation folds 
                (default is 10).
            penalty_cv: Penalty norm used in the penalization 
                (default is "elasticnet").
            penalty: Penalty 
                (default is ['l1', 'l2', 'elasticnet']).
            scoring: Scoring method 
                (default is "accuracy").
            solver_cv: Algorithm to use in the optimization problem 
                (default is "saga").
            solver: Algorithm to use in the optimization problem 
                (default is ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']).
            max_iter_cv: Maximum number of iterations 
                (default is 1000).
            max_iter: Maximum number of iterations 
                (default is [1000]).
            multi_class_cv: Multi-class option 
                (default is "auto").
            multi_class: Multi-class option 
                (default is ['auto', 'ovr', 'multinomial']).
            class_weight_cv: Weights associated with classes 
                (default is None).
            class_weight: Weights associated with classes 
                (default is [None]).
            l1_ratios: List of floats between 0 and 1 passed to ElasticNet 
                (default is [0, 0.1, 0.3, 0.5, 0.7, 0.9]).
            l1_ratio: Ratio of L1 penalty in the "elasticnet" penalty 
                (default is [None]).
        """
        self.X = X
        self.y = y
        self.degree = Val.validate_polynomial_degree(degree)
        self.test_size = Val.validate_test_size(test_size)
        self.random_state = random_state
        self.scaling_first = scaling_first
        self.standard_scaler = Val.validate_bool(standard_scaler)
        self.normal_scaler = Val.validate_bool(normal_scaler)
        self.logistic = Val.validate_bool(logistic)
        self.logistic_cv = Val.validate_bool(logistic_cv)
        self.Cs = Cs
        self.C = C
        self.cv = cv
        self.penalty_cv = penalty_cv
        self.penalty = penalty
        self.scoring = scoring
        self.solver_cv = solver_cv
        self.solver = solver
        self.max_iter_cv = max_iter_cv
        self.max_iter = max_iter
        self.multi_class_cv = multi_class_cv
        self.multi_class = multi_class
        self.class_weight_cv = class_weight_cv
        self.class_weight = class_weight
        self.l1_ratios = l1_ratios
        self.l1_ratio = l1_ratio

        # Train-test split
        self.X_train, self.X_test, \
        self.y_train, self.y_test = \
        train_test_split(self.X, self.y, \
        test_size=self.test_size, random_state=self.random_state)

        if self.scaling_first:

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

        else:
            # Polynomial feature transformation
            self.polynomial_converter = PolynomialFeatures(degree=self.degree,
                                                           include_bias=False)
            self.X_train = self.polynomial_converter.fit_transform(self.X_train)
            self.X_test = self.polynomial_converter.transform(self.X_test)

            if self.standard_scaler:
                self.s_scaler = StandardScaler()
                self.X_train = self.s_scaler.fit_transform(self.X_train)
                self.X_test = self.s_scaler.transform(self.X_test)

            if self.normal_scaler:
                self.n_scaler = Normalizer()
                self.X_train = self.n_scaler.fit_transform(self.X_train)
                self.X_test = self.n_scaler.transform(self.X_test)

        if logistic_cv:
            self.model = LogisticRegressionCV(Cs=self.Cs,
                                              cv=self.cv,
                                              penalty=self.penalty_cv,
                                              scoring=self.scoring,
                                              solver=self.solver_cv,
                                              max_iter=self.max_iter_cv,
                                              multi_class=self.multi_class_cv,
                                              class_weight=self.class_weight_cv,
                                              l1_ratios=self.l1_ratios)
        elif logistic:
            model = LogisticRegression()
            param_grid = {
                "penalty": self.penalty,
                "C": self.C,
                "class_weight": self.class_weight,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "multi_class": self.multi_class,
                "l1_ratio": self.l1_ratio
            }

            self.model = GridSearchCV(estimator=model,
                                      param_grid=param_grid)

        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

    def get_classification_report(self):
        """
        Print classification report.
        """
        print(classification_report(self.y_test, self.y_pred))

    def preparation_lor_func(self,
                             d=None,
                             test_sizes=[0.1, 0.2, 0.3],
                             test_anot=True,
                             train_anot=False):
        """
        Prepare Logistic Regression function for plotting.

        Args:
            d: Range of degrees 
                (default is None).
            test_sizes: List of test sizes to loop over 
                (default is [0.1, 0.2, 0.3]).
            test_anot: Whether to annotate test data 
                (default is True).
            train_anot: Whether to annotate train data 
                (default is False).
        """
        if d is None:
            d = range(1, 7)  # Default range of degrees

        for testsize in test_sizes:
            train_accuracy_list = []
            test_accuracy_list = []

            # Training one model for each selected degree
            for degree in d:
                # Train Test Split
                X_train, X_test, \
                y_train, y_test = train_test_split(self.X,
                                                   self.y,
                                                   test_size=testsize,
                                                   random_state=101)
                # Create Scaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Create Poly-converter
                polynomial_converter = PolynomialFeatures(degree=degree,
                                                        include_bias=False)

                # Create our poly X's
                X_train_poly = polynomial_converter.fit_transform(X_train_scaled)
                X_test_poly = polynomial_converter.transform(X_test_scaled)

                # Train our model
                model = LogisticRegressionCV(fit_intercept=True,
                                            max_iter=10_000)
                model.fit(X=X_train_poly, y=y_train)

                # Calculate and collect accuracy score
                train_pred = model.predict(X_train_poly)
                test_pred = model.predict(X_test_poly)

                train_accuracy = accuracy_score(y_train,train_pred)
                test_accuracy = accuracy_score(y_test,test_pred)

                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)

            plt.figure()  # Creates a new figure for each test size
            plt.plot(d, train_accuracy_list, label="TRAIN", marker="o")
            plt.plot(d, test_accuracy_list, label="TEST", marker="o")

            if test_anot==True:
                for i, txt in enumerate(test_accuracy_list):
                    plt.annotate(f'{txt:.2f}',
                                (d[i],
                                test_accuracy_list[i]),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center')

            if train_anot==True:
                for i, txt in enumerate(train_accuracy_list):
                    plt.annotate(f'{txt:.2f}',
                                (d[i],
                                train_accuracy_list[i]),
                                textcoords="offset points",
                                xytext=(0,10),
                                ha='center')

            plt.title(f"test_size = {testsize}")
            plt.xlabel("Polynomial Complexity")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def count_plot(self, data):
        """
        Plot count plot.

        Args:
            data: Data to plot.
        """
        sns.countplot(data=data,
                      x=self.y,
                      hue=self.y)

    def heat_map(self, data, target_column, annot: bool = True):
        """
        Plot heat map.

        Args:
            data: Dataframe.
            target_column: Target column.
            annot: Whether to annotate 
                (default is True).
        """
        data = pd.get_dummies(data, columns=[target_column], dtype="int8")
        sns.heatmap(data=data.corr(numeric_only=True), annot=annot, cmap="coolwarm")

    def simple_confusion_matrix(self):
        """
        Print simple confusion matrix.
        """
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        print(f"LoR_CV has a confusion matrix of\n{conf_matrix}")

    def confusion_matrix(self, normalize=None):
        """
        Plot confusion matrix.

        Args:
            normalize: Whether to normalize 
                (default is None).
        """
        ConfusionMatrixDisplay.from_estimator(estimator=self.model,
                                              X=self.X_test,
                                              y=self.y_test,
                                              normalize=normalize)

    def AUC_plot(self):
        """
        Plot AUC.
        """
        PrecisionRecallDisplay.from_estimator(estimator=self.model,
                                              X=self.X_test,
                                              y=self.y_test)

    def ROC_plot(self):
        """
        Plot ROC.
        """
        RocCurveDisplay.from_estimator(estimator=self.model,
                                       X=self.X_test,
                                       y=self.y_test)

    @staticmethod
    def plot_3d_scatter(df, x_col, y_col, z_col=None, target_col=None, figsize=(10, 10), dpi=150, alpha=1):
        """
        Plot 3D scatter.

        Args:
            df: Dataframe.
            x_col: X column.
            y_col: Y column.
            z_col: Z column 
                (default is None).
            target_col: Target column.
            figsize: Size of the figure 
                (default is (10, 10)).
            dpi: Dots per inch 
                (default is 150).
            alpha: Alpha (default is 1).
        """
        if not target_col:
            raise ValueError("target_col must be provided")

        fig = plt.figure(figsize=figsize, dpi=dpi, alpha=alpha)
        ax = fig.add_subplot(111, projection="3d")

        # Extract unique classes
        unique_classes = df[target_col].unique()
        num_classes = len(unique_classes)

        # Map classes to integers
        classes_mapping = {classes: i for i, classes in enumerate(unique_classes)}
        colors = df[target_col].map(classes_mapping)

        if z_col:
            # Scatter plot with colors mapped to classes
            sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, cmap='viridis')
            ax.set_zlabel(z_col)
        else:
            # Scatter plot with colors mapped to classes
            sc = ax.scatter(df[x_col], df[y_col], c=colors, cmap='viridis')

        # Add a color bar
        cbar = plt.colorbar(sc, ticks=np.arange(num_classes), label='Classes')
        cbar.set_ticks(np.arange(num_classes))
        cbar.set_ticklabels(unique_classes)

        # Set labels
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.show()

    def plot_multiclass_roc(self, n_classes, figsize=(5, 5)):
        """
        Plot multiclass ROC.

        Args:
            n_classes: Number of classes.
            figsize: Size of the figure 
                (default is (5, 5)).
        """
        y_score = self.model.decision_function(self.X_test)

        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # calculate dummies once
        y_test_dummies = pd.get_dummies(self.y_test, drop_first=False).values
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # roc for each class
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic example')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        plt.show()

    def export_model(self,
                     model_name="final_model",
                     converter_name="final_converter",
                     scaler_name="final_scaler"):
        """
        Export trained model, converter, and scaler.

        Args:
            model_name: Name of the model file 
                (default is "final_model").
            converter_name: Name of the converter file 
                (default is "final_converter").
            scaler_name: Name of the scaler file 
                (default is "final_scaler").
        """
        if self.degree > 1:
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
    def load_model(path: str):
        """
        Load a model from a selected file.

        Args:
            path: Path to the model file.

        Returns:
            Loaded model.
        """
        return load(f"{path}.joblib")
