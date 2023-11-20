#' Instantiates the Xception architecture.
#'
#' @description
#'
#' # Reference
#' - [Xception: Deep Learning with Depthwise Separable Convolutions](
#'     https://arxiv.org/abs/1610.02357) (CVPR 2017)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' The default input image size for this model is 299x299.
#'
#' # Note
#' each Keras Application expects a specific kind of input preprocessing.
#' For Xception, call `keras.applications.xception.preprocess_input`
#' on your inputs before passing them to the model.
#' `xception.preprocess_input` will scale input pixels between -1 and 1.
#'
#' @returns
#'     A model instance.
#'
#' @param include_top
#' whether to include the 3 fully-connected
#' layers at the top of the network.
#'
#' @param weights
#' one of `None` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `False` (otherwise the input shape
#' has to be `(299, 299, 3)`.
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 71.
#' E.g. `(150, 150, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `False`.
#' - `None` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `True`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=True`. Set
#' `classifier_activation=None` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation` can
#' only be `None` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https:/keras.io/api/applications/xception#xception-function>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception>
application_xception <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
}
