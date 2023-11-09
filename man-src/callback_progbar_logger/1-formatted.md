Callback that prints metrics to stdout.

@description

# Raises
    ValueError: In case of invalid `count_mode`.

@param count_mode One of `"steps"` or `"samples"`.
Whether the progress bar should
count samples seen or steps (batches) seen.

@export
@family callback
@seealso
+ <https:/keras.io/api/callbacks/progbar_logger#progbarlogger-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ProgbarLogger>
