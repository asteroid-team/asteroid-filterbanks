import functools
import torch


global SCRIPT_ENABLED
SCRIPT_ENABLED = True


def disable_script_if_tracing():
    global SCRIPT_ENABLED
    SCRIPT_ENABLED = False


def enable_script_if_tracing():
    global SCRIPT_ENABLED
    SCRIPT_ENABLED = True


def is_tracing():
    # Taken for pytorch for compat in 1.6.0
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()


def script_if_tracing(fn):
    # Taken for pytorch for compat in 1.6.0
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Arguments:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing() or not SCRIPT_ENABLED:
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn = torch.jit.script(wrapper.__original_fn)  # type: ignore
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn  # type: ignore
    wrapper.__script_if_tracing_wrapper = True  # type: ignore

    return wrapper
