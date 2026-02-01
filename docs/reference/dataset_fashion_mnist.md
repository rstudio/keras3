# Fashion-MNIST database of fashion articles

Dataset of 60,000 28x28 grayscale images of the 10 fashion article
classes, along with a test set of 10,000 images. This dataset can be
used as a drop-in replacement for MNIST. The class labels are encoded as
integers from 0-9 which correspond to T-shirt/top, Trouser, Pullover,
Dress, Coat, Sandal, Shirt,

## Usage

``` r
dataset_fashion_mnist(convert = TRUE)
```

## Arguments

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`,
where `x` is an array of grayscale image data with shape (num_samples,
28, 28) and `y` is an array of article labels (integers in range 0-9)
with shape (num_samples).

    str(dataset_fashion_mnist())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ y: int [1:60000(1d)] 9 0 0 3 0 2 7 2 5 5 ...
    ##  $ test :List of 2
    ##   ..$ x: int [1:10000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ y: int [1:10000(1d)] 9 2 1 1 6 1 4 6 5 7 ...

    str(dataset_fashion_mnist(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(60000,28,28), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(60000), dtype=uint8>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(10000,28,28), dtype=uint8>
    ##   ..$ y: <numpy.ndarray shape(10000), dtype=uint8>

## Details

Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along
with a test set of 10,000 images. This dataset can be used as a drop-in
replacement for MNIST. The class labels are:

- 0 - T-shirt/top

- 1 - Trouser

- 2 - Pullover

- 3 - Dress

- 4 - Coat

- 5 - Sandal

- 6 - Shirt

- 7 - Sneaker

- 8 - Bag

- 9 - Ankle boot

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/reference/dataset_cifar100.md)  
[`dataset_imdb()`](https://keras3.posit.co/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/reference/dataset_reuters.md)  
