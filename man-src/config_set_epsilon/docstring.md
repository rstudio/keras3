Set the value of the fuzz factor used in numeric expressions.

Args:
    value: float. New value of epsilon.

Examples:
>>> keras.config.epsilon()
1e-07

>>> keras.config.set_epsilon(1e-5)
>>> keras.config.epsilon()
1e-05

>>> # Set it back to the default value.
>>> keras.config.set_epsilon(1e-7)
