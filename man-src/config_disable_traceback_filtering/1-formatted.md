Turn off traceback filtering.

@description
Raw Keras tracebacks (also known as stack traces)
involve many internal frames, which can be
challenging to read through, while not being actionable for end users.
By default, Keras filters internal frames in most exceptions that it
raises, to keep traceback short, readable, and focused on what's
actionable for you (your own code).

See also `keras.config.enable_traceback_filtering()` and
`keras.config.is_traceback_filtering_enabled()`.

If you have previously disabled traceback filtering via
`keras.config.disable_traceback_filtering()`, you can re-enable it via
`keras.config.enable_traceback_filtering()`.

@export
@family traceback utils
@family utils
@family config
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/disable_traceback_filtering>
