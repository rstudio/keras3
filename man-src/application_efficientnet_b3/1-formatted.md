Instantiates the EfficientNetB3 architecture.

@description

# Reference
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
    https://arxiv.org/abs/1905.11946) (ICML 2019)

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
For EfficientNet, input preprocessing is included as part of the model
(as a `Rescaling` layer), and thus
`keras.applications.efficientnet.preprocess_input` is actually a
pass-through function. EfficientNet models expect their inputs to be float
tensors of pixels with values in the `[0-255]` range.

@returns
    A model instance.

@param include_top
Whether to include the fully-connected
layer at the top of the network. Defaults to `True`.

@param weights
One of `None` (random initialization),
`"imagenet"` (pre-training on ImageNet),
or the path to the weights file to be loaded.
Defaults to `"imagenet"`.

@param input_tensor
Optional Keras tensor
(i.e. output of `layers.Input()`)
to use as image input for the model.

@param input_shape
Optional shape tuple, only to be specified
if `include_top` is False.
It should have exactly 3 inputs channels.

@param pooling
Optional pooling mode for feature extraction
when `include_top` is `False`. Defaults to `None`.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
- `avg` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
- `max` means that global max pooling will
    be applied.

@param classes
Optional number of classes to classify images
into, only to be specified if `include_top` is True, and
if no `weights` argument is specified. 1000 is how many
ImageNet classes there are. Defaults to `1000`.

@param classifier_activation
A `str` or callable. The activation function to use
on the "top" layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
Defaults to `'softmax'`.
When loading pretrained weights, `classifier_activation` can only
be `None` or `"softmax"`.

@param ...
Passed on to the Python callable

@export
@seealso
+ <https:/keras.io/api/applications/efficientnet#efficientnetb3-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3>
