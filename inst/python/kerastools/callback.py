from functools import wraps


def wrap_sig_self_idx_logs(fn):
    @wraps(fn)
    def wrapper(self, idx, logs=None):
        res = fn(self, idx + 1, logs)
        if isinstance(res, dict):
            logs.update(res)

    return wrapper


def wrap_sig_self_logs(fn):
    @wraps(fn)
    def wrapper(self, logs=None):
        res = fn(self, logs)
        if isinstance(res, dict):
            logs.update(res)

    return wrapper


def wrap_sig_idx_logs(fn):
    @wraps(fn)
    def wrapper(idx, logs=None):
        res = fn(idx + 1, logs)
        if isinstance(res, dict):
            logs.update(res)

    return wrapper


def wrap_sig_logs(fn):
    @wraps(fn)
    def wrapper(logs=None):
        res = fn(logs)
        if isinstance(res, dict):
            logs.update(res)

    return wrapper
