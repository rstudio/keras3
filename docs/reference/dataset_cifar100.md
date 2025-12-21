# CIFAR100 small image classification

Dataset of 50,000 32x32 color training images, labeled over 100
categories, and 10,000 test images.

## Usage

``` r
dataset_cifar100(label_mode = c("fine", "coarse"), convert = TRUE)
```

## Arguments

- label_mode:

  one of "fine", "coarse".

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`.

    str(dataset_cifar100())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: int [1:50000, 1:32, 1:32, 1:3] 255 255 250 124 43 190 50 178 122 255 ...
    ##   ..$ y: num [1:50000, 1] 19 29 0 11 1 86 90 28 23 31 ...
    ##  $ test :List of 2
    ##   ..$ x: int [1:10000, 1:32, 1:32, 1:3] 199 113 61 93 80 168 37 175 233 182 ...
    ##   ..$ y: num [1:10000, 1] 49 33 72 51 71 92 15 14 23 0 ...

    str(dataset_cifar100(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(50000,32,32,3), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(50000,1), dtype=int64>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(10000,32,32,3), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(10000,1), dtype=int64>

The `x` data is an array of RGB image data with shape (num_samples, 3,
32, 32).

The `y` data is an array of category labels with shape (num_samples).

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/reference/dataset_cifar10.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/reference/dataset_reuters.md)  
