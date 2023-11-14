Computes R2 score.

@description
Formula:

```python
sum_squares_residuals = sum((y_true - y_pred) ** 2)
sum_squares = sum((y_true - mean(y_true)) ** 2)
R2 = 1 - sum_squares_residuals / sum_squares
```

This is also called the
[coefficient of determination](
https://en.wikipedia.org/wiki/Coefficient_of_determination).

It indicates how close the fitted regression line
is to ground-truth data.

- The highest score possible is 1.0. It indicates that the predictors
    perfectly accounts for variation in the target.
- A score of 0.0 indicates that the predictors do not
    account for variation in the target.
- It can also be negative if the model is worse than random.

This metric can also compute the "Adjusted R2" score.

# Examples
```python
y_true = np.array([[1], [4], [3]], dtype=np.float32)
y_pred = np.array([[2], [4], [4]], dtype=np.float32)
metric = keras.metrics.R2Score()
metric.update_state(y_true, y_pred)
result = metric.result()
result
# 0.57142854
```

@param class_aggregation
Specifies how to aggregate scores corresponding to
different output classes (or target dimensions),
i.e. different dimensions on the last axis of the predictions.
Equivalent to `multioutput` argument in Scikit-Learn.
Should be one of
`None` (no aggregation), `"uniform_average"`,
`"variance_weighted_average"`.

@param num_regressors
Number of independent regressors used
("Adjusted R2" score). 0 is the standard R2 score.
Defaults to `0`.

@param name
Optional. string name of the metric instance.

@param dtype
Optional. data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family metric
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/R2Score>
