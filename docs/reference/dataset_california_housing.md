# Loads the California Housing dataset.

This dataset was obtained from the StatLib repository
(`https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html`).

It's a continuous regression dataset with 20,640 samples with 8 features
each.

The target variable is a scalar: the median house value for California
districts, in dollars.

The 8 input features are the following:

- MedInc: median income in block group

- HouseAge: median house age in block group

- AveRooms: average number of rooms per household

- AveBedrms: average number of bedrooms per household

- Population: block group population

- AveOccup: average number of household members

- Latitude: block group latitude

- Longitude: block group longitude

This dataset was derived from the 1990 U.S. census, using one row per
census block group. A block group is the smallest geographical unit for
which the U.S. Census Bureau publishes sample data (a block group
typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the
average number of rooms and bedrooms in this dataset are provided per
household, these columns may take surprisingly large values for block
groups with few households and many empty houses, such as vacation
resorts.

## Usage

``` r
dataset_california_housing(
  version = "large",
  path = "california_housing.npz",
  test_split = 0.2,
  seed = 113L,
  convert = TRUE
)
```

## Arguments

- version:

  `"small"` or `"large"`. The small version contains 600 samples, the
  large version contains 20,640 samples. The purpose of the small
  version is to serve as an approximate replacement for the deprecated
  `boston_housing` dataset.

- path:

  path where to cache the dataset locally (relative to
  `Sys.getenv("KERAS_HOME")`).

- test_split:

  fraction of the data to reserve as test set.

- seed:

  Random seed for shuffling the data before computing the test split.

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Nested list of arrays: `(x_train, y_train), (x_test, y_test)`.

    str(dataset_california_housing())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: num [1:16512, 1:8] -118 -118 -122 -118 -123 ...
    ##   ..$ y: num [1:16512(1d)] 252300 146900 290900 141300 500001 ...
    ##  $ test :List of 2
    ##   ..$ x: num [1:4128, 1:8] -118 -120 -121 -122 -117 ...
    ##   ..$ y: num [1:4128(1d)] 397900 227900 172100 186500 148900 ...

    str(dataset_california_housing(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(16512,8), dtype=float32>
    ##   ..$ y: <numpy.ndarray shape(16512), dtype=float32>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(4128,8), dtype=float32>
    ##   ..$ y: <numpy.ndarray shape(4128), dtype=float32>

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/reference/dataset_boston_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/reference/dataset_reuters.md)  
