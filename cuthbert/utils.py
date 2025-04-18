from typing import Callable, Never, Type


def not_implemented(protocol: Type) -> Callable:
    def f(*args, **kwargs) -> Never:
        raise NotImplementedError(f"{protocol.__name__} not implemented")

    return f
