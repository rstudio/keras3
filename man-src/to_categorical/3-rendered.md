Converts a class vector (integers) to binary class matrix.

@description
E.g. for use with `categorical_crossentropy`.

# Examples

```r
a <- to_categorical(c(0, 1, 2, 3), num_classes=4)
print(a)
```

```
##      [,1] [,2] [,3] [,4]
## [1,]    1    0    0    0
## [2,]    0    1    0    0
## [3,]    0    0    1    0
## [4,]    0    0    0    1
```


```r
b <- array(c(.9, .04, .03, .03,
              .3, .45, .15, .13,
              .04, .01, .94, .05,
              .12, .21, .5, .17),
              dim = c(4, 4))
loss <- k_categorical_crossentropy(a, b)
loss
```

```
## tf.Tensor([0.41284522 0.45601739 0.54430155 0.80437282], shape=(4), dtype=float64)
```


```r
loss <- k_categorical_crossentropy(a, a)
loss
```

```
## tf.Tensor([1.00000005e-07 1.00000005e-07 1.00000005e-07 1.00000005e-07], shape=(4), dtype=float64)
```

@returns
A binary matrix representation of the input as a NumPy array. The class
axis is placed last.

@param x
Array-like with class values to be converted into a matrix
(integers from 0 to `num_classes - 1`).

@param num_classes
Total number of classes. If `NULL`, this would be inferred
as `max(x) + 1`. Defaults to `NULL`.

@export
@family numerical utils
@family utils
@seealso
+ <https:/keras.io/api/utils/python_utils#tocategorical-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical>

