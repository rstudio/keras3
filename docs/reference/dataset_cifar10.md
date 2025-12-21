# CIFAR10 small image classification

Dataset of 50,000 32x32 color training images, labeled over 10
categories, and 10,000 test images.

## Usage

``` r
dataset_cifar10(convert = TRUE)
```

## Arguments

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`.

    str(dataset_cifar10())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: int [1:50000, 1:32, 1:32, 1:3] 59 154 255 28 170 159 164 28 134 125 ...
    ##   ..$ y: int [1:50000, 1] 6 9 9 4 1 1 2 7 8 3 ...
    ##  $ test :List of 2
    ##   ..$ x: int [1:10000, 1:32, 1:32, 1:3] 158 235 158 155 65 179 160 83 23 217 ...
    ##   ..$ y: int [1:10000, 1] 3 8 8 0 6 6 1 6 3 1 ...

    str(dataset_cifar10(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(50000,32,32,3), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(50000,1), dtype=uint8>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(10000,32,32,3), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(10000,1), dtype=uint8>

The `x` data is an array of RGB image data with shape (num_samples, 3,
32, 32).

The `y` data is an array of category labels (integers in range 0-9) with
shape (num_samples).

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/reference/dataset_california_housing.md)  
[`dataset_cifar100()`](https://keras3.posit.co/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/reference/dataset_reuters.md)  
