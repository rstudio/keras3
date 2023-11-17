Instantiates the EfficientNetV2B2 architecture.

@description

# Reference
- [EfficientNetV2: Smaller Models and Faster Training](
    https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples](
https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
https://keras.io/guides/transfer_learning/).

# Note
each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
`keras.applications.efficientnet_v2.preprocess_input` is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the `[0, 255]` range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to `False`.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the `[-1, 1]` range.

@returns
    A model instance.

@param include_top
Boolean, whether to include the fully-connected
layer at the top of the network. Defaults to `True`.

@param weights
One of `None` (random initialization),
`"imagenet"` (pre-training on ImageNet),
or the path to the weights file to be loaded. Defaults to `"imagenet"`.

@param input_tensor
Optional Keras tensor
(i.e. output of `layers.Input()`)
to use as image input for the model.

@param input_shape
Optional shape tuple, only to be specified
if `include_top` is `False`.
It should have exactly 3 inputs channels.

@param pooling
Optional pooling mode for feature extraction
when `include_top` is `False`. Defaults to None.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
- `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
- `"max"` means that global max pooling will
    be applied.

@param classes
Optional number of classes to classify images
into, only to be specified if `include_top` is `True`, and
if no `weights` argument is specified. Defaults to 1000 (number of
ImageNet classes).

@param classifier_activation
A string or callable. The activation function to use
on the "top" layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
Defaults to `"softmax"`.
When loading pretrained weights, `classifier_activation` can only
be `None` or `"softmax"`.

@param include_preprocessing
Boolean, whether to include the preprocessing layer at the bottom of the network.

@export
@seealso
+ <https:/keras.io/api/applications/efficientnet_v2#efficientnetv2b2-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B2>
