keras.metrics.MeanMetricWrapper
__signature__
(
  fn,
  name=None,
  dtype=None,
  **kwargs
)
__doc__
Wrap a stateless metric function with the Mean metric.

You could use this class to quickly build a mean metric from a function. The
function needs to have the signature `fn(y_true, y_pred)` and return a
per-sample loss array. `MeanMetricWrapper.result()` will return
the average metric value across all samples seen so far.

For example:

```python
def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

mse_metric = MeanMetricWrapper(fn=mse)
```

Args:
    fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    **kwargs: Keyword arguments to pass on to `fn`.
