# Preprocesses a tensor or array encoding a batch of images.

Preprocesses a tensor or array encoding a batch of images.

## Usage

``` r
imagenet_preprocess_input(x, data_format = NULL, mode = "caffe")
```

## Arguments

- x:

  Input Numpy or symbolic tensor, 3D or 4D.

- data_format:

  Data format of the image tensor/array.

- mode:

  One of "caffe", "tf", or "torch"

  - caffe: will convert the images from RGB to BGR, then will
    zero-center each color channel with respect to the ImageNet dataset,
    without scaling.

  - tf: will scale pixels between -1 and 1, sample-wise.

  - torch: will scale pixels between 0 and 1 and then will normalize
    each channel with respect to the ImageNet dataset.

## Value

Preprocessed tensor or array.
