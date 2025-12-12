# Instantiates the Inception v3 architecture.

Instantiates the Inception v3 architecture.

## Usage

``` r
application_inception_v3(
  include_top = TRUE,
  weights = "imagenet",
  input_tensor = NULL,
  input_shape = NULL,
  pooling = NULL,
  classes = 1000L,
  classifier_activation = "softmax",
  name = "inception_v3"
)
```

## Arguments

- include_top:

  Boolean, whether to include the fully-connected layer at the top, as
  the last layer of the network. Defaults to `TRUE`.

- weights:

  One of `NULL` (random initialization), `imagenet` (pre-training on
  ImageNet), or the path to the weights file to be loaded. Defaults to
  `"imagenet"`.

- input_tensor:

  Optional Keras tensor (i.e. output of
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md))
  to use as image input for the model. `input_tensor` is useful for
  sharing inputs between multiple different networks. Defaults to
  `NULL`.

- input_shape:

  Optional shape tuple, only to be specified if `include_top` is `FALSE`
  (otherwise the input shape has to be `(299, 299, 3)` (with
  `channels_last` data format) or `(3, 299, 299)` (with `channels_first`
  data format). It should have exactly 3 inputs channels, and width and
  height should be no smaller than 75. E.g. `(150, 150, 3)` would be one
  valid value. `input_shape` will be ignored if the `input_tensor` is
  provided.

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

  optional number of classes to classify images into, only to be
  specified if `include_top` is `TRUE`, and if no `weights` argument is
  specified. Defaults to 1000.

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

- [Rethinking the Inception Architecture for Computer
  Vision](https://arxiv.org/abs/1512.00567) (CVPR 2016)

This function returns a Keras image classification model, optionally
loaded with weights pre-trained on ImageNet.

For image classification use cases, see [this page for detailed
examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the [guide to
transfer learning &
fine-tuning](https://keras.io/guides/transfer_learning/).

## Note

Each Keras Application expects a specific kind of input preprocessing.
For `InceptionV3`, call
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
on your inputs before passing them to the model.
[`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
will scale input pixels between `-1` and `1`.

## See also

- <https://keras.io/api/applications/inceptionv3#inceptionv3-function>
