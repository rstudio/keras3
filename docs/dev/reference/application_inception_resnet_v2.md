# Instantiates the Inception-ResNet v2 architecture.

Instantiates the Inception-ResNet v2 architecture.

## Usage

``` r
application_inception_resnet_v2(
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  input_shape = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = "inception_resnet_v2"
)
```

## Arguments

- include_top:

  whether to include the fully-connected layer at the top of the
  network.

- weights:

  one of `NULL` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded.

- input_tensor:

  optional Keras tensor (i.e. output of
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md))
  to use as image input for the model.

- input_shape:

  optional shape tuple, only to be specified if `include_top` is `FALSE`
  (otherwise the input shape has to be `(299, 299, 3)` (with
  `'channels_last'` data format) or `(3, 299, 299)` (with
  `'channels_first'` data format). It should have exactly 3 inputs
  channels, and width and height should be no smaller than 75. E.g.
  `(150, 150, 3)` would be one valid value.

- pooling:

  Optional pooling mode for feature extraction when `include_top` is
  `FALSE`.

  - `NULL` means that the output of the model will be the 4D tensor
    output of the last convolutional block.

  - `'avg'` means that global average pooling will be applied to the
    output of the last convolutional block, and thus the output of the
    model will be a 2D tensor.

  - `'max'` means that global max pooling will be applied.

- classes:

  optional number of classes to classify images into, only to be
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

A model instance.

## Reference

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections
  on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

This function returns a Keras image classification model, optionally
loaded with weights pre-trained on ImageNet.

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For `InceptionResNetV2`, call
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
on your inputs before passing them to the model.
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
will scale input pixels between -1 and 1.

## See also

- <https://keras.io/api/applications/inceptionresnetv2#inceptionresnetv2-function>
