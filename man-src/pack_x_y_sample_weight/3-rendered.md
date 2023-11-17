Packs user-provided data into a list.

@description
This is a convenience utility for packing data into the list formats
that `fit()` uses.

# Usage
Standalone usage:


```r
x <- k_ones(c(10, 1))
data <- pack_x_y_sample_weight(x)

# TRUE
y <- k_ones(c(10, 1))
data <- pack_x_y_sample_weight(x, y)
```

@returns
    List in the format used in `fit()`.

@param x
Features to pass to `Model`.

@param y
Ground-truth targets to pass to `Model`.

@param sample_weight
Sample weight for each element.

@export
@family datum util adapter trainers
@family datum adapter trainers
@family trainers
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pack_x_y_sample_weight>

