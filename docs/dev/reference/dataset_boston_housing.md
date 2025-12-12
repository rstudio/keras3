# Boston housing price regression dataset

Dataset taken from the StatLib library which is maintained at Carnegie
Mellon University.

## Usage

``` r
dataset_boston_housing(
  path = "boston_housing.npz",
  test_split = 0.2,
  seed = 113L,
  convert = TRUE
)
```

## Arguments

- path:

  Path where to cache the dataset locally (relative to
  ~/.keras/datasets).

- test_split:

  fraction of the data to reserve as test set.

- seed:

  Random seed for shuffling the data before computing the test split.

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`.

Samples contain 13 attributes of houses at different locations around
the Boston suburbs in the late 1970s. Targets are the median values of
the houses at a location (in k\$).

    str(dataset_boston_housing())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: num [1:404, 1:13] 1.2325 0.0218 4.8982 0.0396 3.6931 ...
    ##   ..$ y: num [1:404(1d)] 15.2 42.3 50 21.1 17.7 18.5 11.3 15.6 15.6 14.4 ...
    ##  $ test :List of 2
    ##   ..$ x: num [1:102, 1:13] 18.0846 0.1233 0.055 1.2735 0.0715 ...
    ##   ..$ y: num [1:102(1d)] 7.2 18.8 19 27 22.2 24.5 31.2 22.9 20.5 23.2 ...

    str(dataset_boston_housing(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(404,13), dtype=float64>
    ##   ..$ y: <numpy.ndarray shape(404), dtype=float64>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(102,13), dtype=float64>
    ##   ..$ y: <numpy.ndarray shape(102), dtype=float64>

## See also

Other datasets:  
[`dataset_california_housing()`](https://keras3.posit.co/dev/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/dev/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/dev/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/dev/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/dev/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/dev/reference/dataset_reuters.md)  
