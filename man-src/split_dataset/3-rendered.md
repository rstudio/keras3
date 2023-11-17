Splits a dataset into a left half and a right half (e.g. train / test).

@description

# Examples

```r
data <- random_uniform(c(1000, 4))
c(left_ds, right_ds) %<-% split_dataset(list(data$numpy()), left_size = 0.8)
left_ds$cardinality()
```

```
## tf.Tensor(800, shape=(), dtype=int64)
```

```r
right_ds$cardinality()
```

```
## tf.Tensor(200, shape=(), dtype=int64)
```

@returns
A list of two `tf$data$Dataset` objects:
the left and right splits.

@param dataset
A `tf$data$Dataset`, a `torch$utils$data$Dataset` object,
or a list of arrays with the same length.

@param left_size
If float (in the range `[0, 1]`), it signifies
the fraction of the data to pack in the left dataset. If integer, it
signifies the number of samples to pack in the left dataset. If
`NULL`, defaults to the complement to `right_size`.
Defaults to `NULL`.

@param right_size
If float (in the range `[0, 1]`), it signifies
the fraction of the data to pack in the right dataset.
If integer, it signifies the number of samples to pack
in the right dataset.
If `NULL`, defaults to the complement to `left_size`.
Defaults to `NULL`.

@param shuffle
Boolean, whether to shuffle the data before splitting it.

@param seed
A random seed for shuffling.

@export
@family dataset utils
@family utils
@seealso
+ <https:/keras.io/api/utils/python_utils#splitdataset-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset>

