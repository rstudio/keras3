

# import keras ## imported from R to keep it consistant

from abc import ABCMeta

def keras_class_type(x):
  if not isinstance(x, (type, ABCMeta)):
    return ""
  if issubclass(x, keras.layers.Layer):
    return "layer"
  if issubclass(x, keras.callbacks.Callback):
    return "callback"
  if issubclass(x, keras.constraints.Constraint):
    return "constraint"
  if issubclass(x, keras.initializers.Initializer):
    return "initializer"
  if issubclass(x, keras.optimizers.schedules.LearningRateSchedule):
    return "learning_rate_schedule"
  if issubclass(x, keras.optimizers.Optimizer):
    return "optimizer"
  if issubclass(x, keras.losses.Loss):
    return "loss"
  if issubclass(x, keras.metrics.Metric):
    return "metric"
  return ""



import inspect

def py_get_class_from_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)  # handle special descriptor objects


def py_ismethod(function):
    return py_get_class_from_method(function) is not None

# import keras_cv
# import keras_nlp
  # if issubclass(x, keras.):
  #   return "Activation"
  # if issubclass(x, keras.applications):
  #   return "Application"
