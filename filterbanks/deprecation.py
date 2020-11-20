import warnings
from functools import wraps


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """

    # Taken from numpy


def mark_deprecated(message, version=None):
    """Decorator to add deprecation message.

    Args:
        message: Migration steps to be given to users.
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            from_what = "a future release" if version is None else f"asteroid v{version}"
            warn_message = (
                f"{func.__module__}.{func.__name__} has been deprecated "
                f"and will be removed from {from_what}. "
                f"{message}"
            )
            warnings.warn(warn_message, VisibleDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator
