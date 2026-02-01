# Instantiates the EfficientNetV2B3 architecture.

Instantiates the EfficientNetV2B3 architecture.

## Usage

``` r
application_efficientnet_v2b3(
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  input_shape = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  include_preprocessing = TRUE,
  name = "efficientnetv2-b3"
)
```

## Arguments

- include_top:

  Boolean, whether to include the fully-connected layer at the top of
  the network. Defaults to `TRUE`.

- weights:

  One of `NULL` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded. Defaults to
  `"imagenet"`.

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

  - `"avg"` means that global average pooling will be applied to the
    output of the last convolutional layer, and thus the output of the
    model will be a 2D tensor.

  - `"max"` means that global max pooling will be applied.

- classes:

  Optional number of classes to classify images into, only to be
  specified if `include_top` is `TRUE`, and if no `weights` argument is
  specified. Defaults to 1000 (number of ImageNet classes).

- classifier_activation:

  A string or callable. The activation function to use on the "top"
  layer. Ignored unless `include_top=TRUE`. Set
  `classifier_activation=NULL` to return the logits of the "top" layer.
  Defaults to `"softmax"`. When loading pretrained weights,
  `classifier_activation` can only be `NULL` or `"softmax"`.

- include_preprocessing:

  Boolean, whether to include the preprocessing layer at the bottom of
  the network.

- name:

  The name of the model (string).

## Value

A model instance.

## Reference

- [EfficientNetV2: Smaller Models and Faster
  Training](https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model, optionally
loaded with weights pre-trained on ImageNet.

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part
of the model (as a `Rescaling` layer), and thus
[`application_preprocess_inputs()`](https://keras3.posit.co/reference/process_utils.md)
is actually a pass-through function. In this use case, EfficientNetV2
models expect their inputs to be float tensors of pixels with values in
the `[0, 255]` range. At the same time, preprocessing as a part of the
model (i.e. `Rescaling` layer) can be disabled by setting
`include_preprocessing` argument to `FALSE`. With preprocessing disabled
EfficientNetV2 models expect their inputs to be float tensors of pixels
with values in the `[-1, 1]` range.

## See also

- <https://keras.io/api/applications/efficientnet_v2#efficientnetv2b3-function>
