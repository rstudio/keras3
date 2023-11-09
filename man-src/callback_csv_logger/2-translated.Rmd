Callback that streams epoch results to a CSV file.

@description
Supports all values that can be represented as a string,
including 1D iterables such as atomic vectors.

# Examples
```r
csv_logger <- callback_csv_logger('training.log')
model %>% fit(X_train, Y_train, callbacks = list(csv_logger))
```

@param filename Filename of the CSV file, e.g. `'run/log.csv'`.
@param separator String used to separate elements in the CSV file.
@param append Boolean. `TRUE`: append if file exists (useful for continuing
    training). `FALSE`: overwrite existing file.

@export
@family callback
@seealso
+ <https:/keras.io/api/callbacks/csv_logger#csvlogger-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger>
