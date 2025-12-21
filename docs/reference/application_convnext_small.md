# Instantiates the ConvNeXtSmall architecture.

Instantiates the ConvNeXtSmall architecture.

## Usage

``` r
application_convnext_small(
  include_top = TRUE,
  include_preprocessing = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  input_shape = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = "convnext_small"
)
```

## Arguments

- include_top:

  Whether to include the fully-connected layer at the top of the
  network. Defaults to `TRUE`.

- include_preprocessing:

  Boolean, whether to include the preprocessing layer at the bottom of
  the network.

- weights:

  One of `NULL` (random initialization), `"imagenet"` (pre-training on
  ImageNet-1k), or the path to the weights file to be loaded. Defaults
  to `"imagenet"`.

- input_tensor:

  Optional Keras tensor (i.e. output of
  [`keras_input()`](https://keras3.posit.co/reference/keras_input.md))
  to use as image input for the model.

- input_shape:

  Optional shape tuple, only to be specified if `include_top` is
  `FALSE`. It should have exactly 3 inputs channels.

- pooling:

  Optional pooling mode for feature extraction when `include_top` is
  `FALSE`. Defaults to `NULL`.

  - `NULL` means that the output of the model will be the 4D tensor
    output of the last convolutional layer.

  - `avg` means that global average pooling will be applied to the
    output of the last convolutional layer, and thus the output of the
    model will be a 2D tensor.

  - `max` means that global max pooling will be applied.

- classes:

  Optional number of classes to classify images into, only to be
  specified if `include_top` is `TRUE`, and if no `weights` argument is
  specified. Defaults to 1000 (number of ImageNet classes).

- classifier_activation:

  A `str` or callable. The activation function to use on the "top"
  layer. Ignored unless `include_top=TRUE`. Set
  `classifier_activation=NULL` to return the logits of the "top" layer.
  Defaults to `"softmax"`. When loading pretrained weights,
  `classifier_activation` can only be `NULL` or `"softmax"`.

- name:

  The name of the model (string).

## Value

A model instance.

## References

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (CVPR
  2022)

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).
For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

The `base`, `large`, and `xlarge` models were first pre-trained on the
ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
pre-trained parameters of the models were assembled from the [official
repository](https://github.com/facebookresearch/ConvNeXt). To get a
sense of how these parameters were converted to Keras compatible
parameters, please refer to [this
repository](https://github.com/sayakpaul/keras-convnext-conversion).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For ConvNeXt, preprocessing is included in the model using a
`Normalization` layer. ConvNeXt models expect their inputs to be float
or uint8 tensors of pixels with values in the `[0-255]` range.

When calling the [`summary()`](https://rdrr.io/r/base/summary.html)
method after instantiating a ConvNeXt model, prefer setting the
`expand_nested` argument
[`summary()`](https://rdrr.io/r/base/summary.html) to `TRUE` to better
investigate the instantiated model.

## See also

- <https://keras.io/api/applications/convnext#convnextsmall-function>
