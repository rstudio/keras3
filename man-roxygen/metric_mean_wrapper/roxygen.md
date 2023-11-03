Wrap a stateless metric function with the Mean metric.

@description
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

@param fn The metric function to wrap, with signature
    `fn(y_true, y_pred, **kwargs)`.
@param name (Optional) string name of the metric instance.
@param dtype (Optional) data type of the metric result.
@param ... Keyword arguments to pass on to `fn`.

@export
@family metric
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanMetricWrapper>
