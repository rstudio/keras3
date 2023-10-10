

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

  # if issubclass(x, keras.):
  #   return "Activation"
  # if issubclass(x, keras.applications):
  #   return "Application"
