Converts a class vector (integers) to binary class matrix.

@description
E.g. for use with `categorical_crossentropy`.

# Examples
```python
a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
print(a)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

```python
b = np.array([.9, .04, .03, .03,
              .3, .45, .15, .13,
              .04, .01, .94, .05,
              .12, .21, .5, .17],
              shape=[4, 4])
loss = keras.backend.categorical_crossentropy(a, b)
print(np.around(loss, 5))
# [0.10536 0.82807 0.1011  1.77196]
```

```python
loss = keras.backend.categorical_crossentropy(a, a)
print(np.around(loss, 5))
# [0. 0. 0. 0.]
```

@returns
A binary matrix representation of the input as a NumPy array. The class
axis is placed last.

@param x Array-like with class values to be converted into a matrix
    (integers from 0 to `num_classes - 1`).
@param num_classes Total number of classes. If `None`, this would be inferred
    as `max(x) + 1`. Defaults to `None`.

@export
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical>
