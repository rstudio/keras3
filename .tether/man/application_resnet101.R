#' Instantiates the ResNet101 architecture.
#'
#' @description
#'
#' # Reference
#' - [Deep Residual Learning for Image Recognition](
#'     https://arxiv.org/abs/1512.03385) (CVPR 2015)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call `keras.applications.resnet.preprocess_input` on your
#' inputs before passing them to the model. `resnet.preprocess_input` will convert
#' the input images from RGB to BGR, then will zero-center each color channel with
#' respect to the ImageNet dataset, without scaling.
#'
#' @returns
#'     A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `None` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `False` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `False`.
#' - `None` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `True`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=True`. Set
#' `classifier_activation=None` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `None` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https:/keras.io/api/applications/resnet#resnet101-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101>
application_resnet101 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
}
