class Validation():
    """
    Class for storing validation method.
    """

    @staticmethod
    def validate_polynomial_degree(value):
        """
        Validation method to validate that the input for polynomial degree
        is valid.
        """
        if value is True:
            raise ValueError("True is not a valid input for degree")
        if isinstance(value, int) and value > 0:
            return value
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
            if all(v > 0 for v in value):
                return value
            else:
                raise ValueError("Tuple values must be positive integers")
        else:
            raise ValueError("Value must be a positive int or a tuple of two positive ints")

    @staticmethod
    def validate_bool(value):
        """
        Validation method to validate that input is of type bool.
        """
        if isinstance(value, bool):
            return value
        else:
            raise ValueError("Value must be of bool type")

    @staticmethod
    def validate_test_size(value: float):
        """
        Validation method for test_size value in the train_test_split
        method.
        """
        if isinstance(value, float) and 0.0 < value < 1.0:
            return value
        else:
            raise ValueError("Value must be a float between 0.0 and 1.0")

    @staticmethod
    def validate_alphas(value):
        """
        Validation method for alpha values in some models.
        """
        if isinstance(value, bool):
            raise ValueError("Boolean values are not accepted")
        elif isinstance(value, tuple):
            if all((isinstance(v, (int, float)) and v > 0 and not isinstance(v, bool)) for v in value):
                return value
            else:
                raise ValueError("Tuple values must be positive values")
        elif isinstance(value, (int, float)) and value > 0:
            return value
        else:
            raise ValueError("Value must be a positive value, a tuple of two positive values, or a positive float")

    def validate_not_bool(value): # pylint: disable=no-self-argument
        """
        Validation method to check that input isn't a bool.
        """
        if isinstance(value, bool):
            raise ValueError("Boolean values are not accepted")
        else:
            return value
