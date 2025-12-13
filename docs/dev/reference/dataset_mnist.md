# MNIST database of handwritten digits

Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a
test set of 10,000 images.

## Usage

``` r
dataset_mnist(path = "mnist.npz", convert = TRUE)
```

## Arguments

- path:

  Path where to cache the dataset locally (relative to
  ~/.keras/datasets).

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`,
where `x` is an array of grayscale image data with shape (num_samples,
28, 28) and `y` is an array of digit labels (integers in range 0-9) with
shape (num_samples).

    str(dataset_mnist())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ y: int [1:60000(1d)] 5 0 4 1 9 2 1 3 1 4 ...
    ##  $ test :List of 2
    ##   ..$ x: int [1:10000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ y: int [1:10000(1d)] 7 2 1 0 4 1 4 9 5 9 ...

    str(dataset_mnist(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(60000,28,28), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(60000), dtype=uint8>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(10000,28,28), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(10000), dtype=uint8>

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/dev/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/dev/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/dev/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/dev/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/dev/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)  
[`dataset_reuters()`](https://keras3.posit.co/dev/reference/dataset_reuters.md)  
