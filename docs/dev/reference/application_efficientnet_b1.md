# Instantiates the EfficientNetB1 architecture.

Instantiates the EfficientNetB1 architecture.

## Usage

``` r
application_efficientnet_b1(
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  input_shape = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = "efficientnetb1",
  ...
)
```

## Arguments

- include_top:

  Whether to include the fully-connected layer at the top of the
  network. Defaults to `TRUE`.

- weights:

  One of `NULL` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded. Defaults to
  `"imagenet"`.

- input_tensor:

  Optional Keras tensor (i.e. output of
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md))
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
  specified if `include_top` is TRUE, and if no `weights` argument is
  specified. 1000 is how many ImageNet classes there are. Defaults to
  `1000`.

- classifier_activation:

  A `str` or callable. The activation function to use on the "top"
  layer. Ignored unless `include_top=TRUE`. Set
  `classifier_activation=NULL` to return the logits of the "top" layer.
  Defaults to `'softmax'`. When loading pretrained weights,
  `classifier_activation` can only be `NULL` or `"softmax"`.

- name:

  The name of the model (string).

- ...:

  For forward/backward compatability.

## Value

A model instance.

## Reference

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

This function returns a Keras image classification model, optionally
loaded with weights pre-trained on ImageNet.

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
is actually a pass-through function. EfficientNet models expect their
inputs to be float tensors of pixels with values in the `[0-255]` range.

## See also

- <https://keras.io/api/applications/efficientnet#efficientnetb1-function>
