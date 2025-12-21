# Instantiates the MobileNet architecture.

Instantiates the MobileNet architecture.

## Usage

``` r
application_mobilenet(
  input_shape = NULL,
  alpha = 1,
  depth_multiplier = 1L,
  dropout = 0.001,
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = NULL
)
```

## Arguments

- input_shape:

  Optional shape tuple, only to be specified if `include_top` is `FALSE`
  (otherwise the input shape has to be `(224, 224, 3)` (with
  `"channels_last"` data format) or `(3, 224, 224)` (with
  `"channels_first"` data format). It should have exactly 3 inputs
  channels, and width and height should be no smaller than 32. E.g.
  `(200, 200, 3)` would be one valid value. Defaults to `NULL`.
  `input_shape` will be ignored if the `input_tensor` is provided.

- alpha:

  Controls the width of the network. This is known as the width
  multiplier in the MobileNet paper.

  - If `alpha < 1.0`, proportionally decreases the number of filters in
    each layer.

  - If `alpha > 1.0`, proportionally increases the number of filters in
    each layer.

  - If `alpha == 1`, default number of filters from the paper are used
    at each layer. Defaults to `1.0`.

- depth_multiplier:

  Depth multiplier for depthwise convolution. This is called the
  resolution multiplier in the MobileNet paper. Defaults to `1.0`.

- dropout:

  Dropout rate. Defaults to `0.001`.

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
  to use as image input for the model. `input_tensor` is useful for
  sharing inputs between multiple different networks. Defaults to
  `NULL`.

- pooling:

  Optional pooling mode for feature extraction when `include_top` is
  `FALSE`.

  - `NULL` (default) means that the output of the model will be the 4D
    tensor output of the last convolutional block.

  - `avg` means that global average pooling will be applied to the
    output of the last convolutional block, and thus the output of the
    model will be a 2D tensor.

  - `max` means that global max pooling will be applied.

- classes:

  Optional number of classes to classify images into, only to be
  specified if `include_top` is `TRUE`, and if no `weights` argument is
  specified. Defaults to `1000`.

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

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
  Applications](https://arxiv.org/abs/1704.04861)

This function returns a Keras image classification model, optionally
loaded with weights pre-trained on ImageNet.

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For MobileNet, call
[`application_preprocess_inputs()`](https://keras3.posit.co/reference/process_utils.md)
on your inputs before passing them to the model.
[`application_preprocess_inputs()`](https://keras3.posit.co/reference/process_utils.md)
will scale input pixels between `-1` and `1`.

## See also

- <https://keras.io/api/applications/mobilenet#mobilenet-function>
