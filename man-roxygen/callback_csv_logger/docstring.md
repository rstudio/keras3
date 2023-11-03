Callback that streams epoch results to a CSV file.

Supports all values that can be represented as a string,
including 1D iterables such as `np.ndarray`.

Args:
    filename: Filename of the CSV file, e.g. `'run/log.csv'`.
    separator: String used to separate elements in the CSV file.
    append: Boolean. True: append if file exists (useful for continuing
        training). False: overwrite existing file.

Example:

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```
