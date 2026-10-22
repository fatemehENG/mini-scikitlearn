import inspect

class BaseEstimator:
    def __init__(self):
        pass

    def get_params(self):

        signature = inspect.signature(self.__init__)

        params = {}

        for name in signature.parameters:

            if name != "self":
                params[name] = getattr(self, name)

        return params


    def set_params(self, **params):
        valid_params = self.get_params()

        for name, value in params.items():

            if name not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{name}' for {self.__class__.__name__}"
                )

            setattr(self, name, value)

        return self








