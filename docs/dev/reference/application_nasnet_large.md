# Instantiates a NASNet model in ImageNet mode.

Instantiates a NASNet model in ImageNet mode.

## Usage

``` r
application_nasnet_large(
  input_shape = NULL,
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = "nasnet_large"
)
```

## Arguments

- input_shape:

  Optional shape tuple, only to be specified if `include_top` is `FALSE`
  (otherwise the input shape has to be `(331, 331, 3)` for NASNetLarge.
  It should have exactly 3 inputs channels, and width and height should
  be no smaller than 32. E.g. `(224, 224, 3)` would be one valid value.

- include_top:

  Whether to include the fully-connected layer at the top of the
  network.

- weights:

  `NULL` (random initialization) or `imagenet` (ImageNet weights). For
  loading `imagenet` weights, `input_shape` should be (331, 331, 3)

- input_tensor:

  Optional Keras tensor (i.e. output of
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md))
  to use as image input for the model.

- pooling:

  Optional pooling mode for feature extraction when `include_top` is
  `FALSE`.

  - `NULL` means that the output of the model will be the 4D tensor
    output of the last convolutional layer.

  - `avg` means that global average pooling will be applied to the
    output of the last convolutional layer, and thus the output of the
    model will be a 2D tensor.

  - `max` means that global max pooling will be applied.

- classes:

  Optional number of classes to classify images into, only to be
  specified if `include_top` is `TRUE`, and if no `weights` argument is
  specified.

- classifier_activation:

  A `str` or callable. The activation function to use on the "top"
  layer. Ignored unless `include_top=TRUE`. Set
  `classifier_activation=NULL` to return the logits of the "top" layer.
  When loading pretrained weights, `classifier_activation` can only be
  `NULL` or `"softmax"`.

- name:

  The name of the model (string).

## Value

A Keras model instance.

## Reference

- [Learning Transferable Architectures for Scalable Image
  Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)

Optionally loads weights pre-trained on ImageNet. Note that the data
format convention used by the model is the one specified in your Keras
config at `~/.keras/keras.json`.

## Note

Each Keras Application expects a specific kind of input preprocessing.
For NASNet, call
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
on your inputs before passing them to the model.

## See also

- <https://keras.io/api/applications/nasnet#nasnetlarge-function>
