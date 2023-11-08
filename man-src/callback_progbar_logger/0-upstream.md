keras.callbacks.ProgbarLogger
__signature__
()
__doc__
Callback that prints metrics to stdout.

Args:
    count_mode: One of `"steps"` or `"samples"`.
        Whether the progress bar should
        count samples seen or steps (batches) seen.

Raises:
    ValueError: In case of invalid `count_mode`.
