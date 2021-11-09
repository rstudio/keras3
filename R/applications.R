
#' Instantiates the Xception architecture
#'
#' @details
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
#' @note
#' Each Keras Application typically expects a specific kind of input preprocessing.
#' For Xception, call `xception_preprocess_input()` on your
#' inputs before passing them to the model.
#' `xception_preprocess_input()` will scale input pixels between -1 and 1.
#'
#' @section
#' Reference:
#' - [Xception: Deep Learning with Depthwise Separable Convolutions](
#'     https://arxiv.org/abs/1610.02357) (CVPR 2017)
#'
#' @param include_top Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights One of `NULL` (random initialization),
#' `'imagenet'` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `'imagenet'`.
#'
#' @param input_tensor Optional Keras tensor
#' (i.e. output of `layer_input()`)
#' to use as image input for the model.
#'
#' @param input_shape optional shape list, only to be specified
#' if `include_top` is FALSE (otherwise the input shape
#' has to be `(299, 299, 3)`.
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 71.
#' E.g. `(150, 150, 3)` would be one valid value.
#'
#' @param pooling Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `'avg'` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `'max'` means that global max pooling will
#'     be applied.
#'
#' @param classes Optional number of classes to classify images into, only to be
#'   specified if `include_top` is TRUE, and if no `weights` argument is
#'   specified. Defaults to 1000 (number of ImageNet classes).
#'
#' @param classifier_activation A string or callable. The activation function to
#'   use on the "top" layer. Ignored unless `include_top = TRUE`. Set
#'   `classifier_activation = NULL` to return the logits of the "top" layer.
#'   Defaults to `'softmax'`. When loading pretrained weights,
#'   `classifier_activation` can only be `NULL` or `"softmax"`.
#'
#' @param ... For backwards and forwards compatibility
#'
#'
#' @param x `preprocess_input()` takes an array or floating point tensor, 3D or
#'   4D with 3 color channels, with values in the range `[0, 255]`.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception>
#'   +  <https://keras.io/api/applications/>
#'
#'
#' @export
application_xception <-
function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
         input_shape = NULL, pooling = NULL, classes = 1000,
         classifier_activation='softmax', ...)
{
  verify_application_prerequistes()
  args <- capture_args(match.call(), list(
    classes = as.integer,
    input_shape = normalize_shape))
  do.call(keras$applications$Xception, args)
}


#' @rdname application_xception
#' @export
xception_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$xception$preprocess_input)
}


#' VGG16 and VGG19 models for Keras.
#'
#' @details Optionally loads weights pre-trained on ImageNet.
#'
#' The `imagenet_preprocess_input()` function should be used for image preprocessing.
#'
#' @inheritParams application_xception
#'
#' @param include_top whether to include the 3 fully-connected layers at the top
#'   of the network.
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(224, 224, 3)` It should
#'   have exactly 3 inputs channels, and width and height should be no smaller
#'   than 32. E.g. `(200, 200, 3)` would be one valid value.
#'
#' @return Keras model instance.
#'
#' @section Reference: - [Very Deep Convolutional Networks for Large-Scale Image
#'   Recognition](https://arxiv.org/abs/1409.1556)
#'
#' @name application_vgg
#'
#' @examples
#' \dontrun{
#' library(keras)
#'
#' model <- application_vgg16(weights = 'imagenet', include_top = FALSE)
#'
#' img_path <- "elephant.jpg"
#' img <- image_load(img_path, target_size = c(224,224))
#' x <- image_to_array(img)
#' x <- array_reshape(x, c(1, dim(x)))
#' x <- imagenet_preprocess_input(x)
#'
#' features <- model %>% predict(x)
#' }
#' @export
application_vgg16 <-
function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
         input_shape = NULL, pooling = NULL, classes = 1000,
         classifier_activation='softmax')
{
  verify_application_prerequistes()
  args <- capture_args(match.call(), list(
      classes = as.integer,
      input_shape = normalize_shape))
  do.call(keras$applications$VGG16, args)
}

#' @rdname application_vgg
#' @export
application_vgg19 <-
function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
         input_shape = NULL, pooling = NULL, classes = 1000,
         classifier_activation='softmax')
{
  verify_application_prerequistes()
  args <- capture_args(match.call(), list(
      classes = as.integer,
      input_shape = normalize_shape))
  do.call(keras$applications$VGG19, args)
}


#' Instantiates the ResNet architecture
#'
#' @details
#' Reference:
#' - [Deep Residual Learning for Image Recognition](
#'     https://arxiv.org/abs/1512.03385) (CVPR 2015)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' Note: each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call `tf.keras.applications.resnet.preprocess_input` on your
#' inputs before passing them to the model.
#' `resnet.preprocess_input` will convert the input images from RGB to BGR,
#' then will zero-center each color channel with respect to the ImageNet dataset,
#' without scaling.
#'
#' @inheritParams application_efficientnet
#'
#' @param input_shape optional shape list, only to be specified
#' if `include_top` is FALSE (otherwise the input shape
#' has to be `c(224, 224, 3)` (with `'channels_last'` data format)
#' or `c(3, 224, 224)` (with `'channels_first'` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `c(200, 200, 3)` would be one valid value.
#'
#' @param x `preprocess_input()` takes an array or floating point tensor, 3D or
#'   4D with 3 color channels, with values in the range `[0, 255]`.
#'
#' @param ... For backwards and forwards compatibility
#'
#' @name application_resnet
#' @rdname application_resnet
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/ResNet101>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/ResNet152>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet101V2>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet152V2>
#'   +  <https://keras.io/api/applications/>
#'
#' @examples
#' \dontrun{
#' library(keras)
#'
#' # instantiate the model
#' model <- application_resnet50(weights = 'imagenet')
#'
#' # load the image
#' img_path <- "elephant.jpg"
#' img <- image_load(img_path, target_size = c(224,224))
#' x <- image_to_array(img)
#'
#' # ensure we have a 4d tensor with single element in the batch dimension,
#' # the preprocess the input for prediction using resnet50
#' x <- array_reshape(x, c(1, dim(x)))
#' x <- imagenet_preprocess_input(x)
#'
#' # make predictions then decode and print them
#' preds <- model %>% predict(x)
#' imagenet_decode_predictions(preds, top = 3)[[1]]
#' }
NULL

## TODO: maybe expand all the application wrappers to use this?
## then clean up with `formals(fn)$classifier_activation <- NULL` where needed
new_application_resnet_wrapper <- function(name) {
  args <- alist(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
                input_shape = NULL, pooling = NULL, classes = 1000)
  if(grepl("V2$", name))
    args <- c(args, alist(classifier_activation='softmax'))
  args <- c(args, alist(... = ))

  body <- substitute({
    args <- capture_args(match.call(), list(
      classes = as.integer,
      input_shape = normalize_shape))
    do.call(keras$applications$NAME, args)
  }, list(NAME = name))

  as.function(c(args, body), envir = parent.frame())
}

#' @export
#' @rdname application_resnet
application_resnet50  <- new_application_resnet_wrapper("ResNet50")

#' @export
#' @rdname application_resnet
application_resnet101 <- new_application_resnet_wrapper("ResNet101")

#' @export
#' @rdname application_resnet
application_resnet152 <- new_application_resnet_wrapper("ResNet152")

#' @export
#' @rdname application_resnet
application_resnet50_v2  <- new_application_resnet_wrapper("ResNet50V2")

#' @export
#' @rdname application_resnet
application_resnet101_v2 <- new_application_resnet_wrapper("ResNet101V2")

#' @export
#' @rdname application_resnet
application_resnet152_v2 <- new_application_resnet_wrapper("ResNet152V2")


#' @export
#' @rdname application_resnet
resnet_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$resnet$preprocess_input)
}

#' @export
#' @rdname application_resnet
resnet_v2_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$resnet_v2$preprocess_input)
}





#' Inception V3 model, with weights pre-trained on ImageNet.
#'
#' @details
#' Do note that the input image format for this model is different than for
#' the VGG16 and ResNet models (299x299 instead of 224x224).
#'
#' The `inception_v3_preprocess_input()` function should be used for image
#' preprocessing.
#'
#' @inheritParams application_xception
#'
#' @return A Keras model instance.
#'
#' @section Reference:
#'  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
#'
#' @export
application_inception_v3 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
                                     pooling = NULL, classes = 1000,  classifier_activation='softmax', ...) {
  verify_application_prerequistes()
  args <- capture_args(match.call(), list(
    input_shape = normalize_shape, classes = as.integer))
  do.call(keras$applications$InceptionV3, args)
}


#' @rdname application_inception_v3
#' @export
inception_v3_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$inception_v3$preprocess_input)
}


#' Inception-ResNet v2 model, with weights trained on ImageNet
#'
#'
#' @inheritParams application_xception
#'
#' @return A Keras model instance.
#'
#' @details
#' Do note that the input image format for this model is different than for
#' the VGG16 and ResNet models (299x299 instead of 224x224).
#'
#' The `inception_resnet_v2_preprocess_input()` function should be used for image
#' preprocessing.
#'
#' @section Reference:
#'  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)(https://arxiv.org/abs/1512.00567)
#'
#' @export
application_inception_resnet_v2 <-
function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000, classifier_activation='softmax', ...) {
  verify_application_prerequistes()
  args <- capture_args(match.call(), list(
    input_shape = normalize_shape, classes = as.integer))
  do.call(keras$applications$InceptionResNetV2, args)
}

#' @rdname application_inception_resnet_v2
#' @export
inception_resnet_v2_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$inception_resnet_v2$preprocess_input)
}

#' Decodes the prediction of an ImageNet model.
#'
#' @param preds Tensor encoding a batch of predictions.
#' @param top integer, how many top-guesses to return.
#'
#' @return List of data frames with variables `class_name`, `class_description`,
#'   and `score` (one data frame per sample in batch input).
#'
#' @export
imagenet_decode_predictions <- function(preds, top = 5) {

  # decode predictions
  # we use the vgg16 function which is the same as imagenet_utils
  decoded <- keras$applications$vgg16$decode_predictions(
    preds = preds,
    top = as.integer(top)
  )

  # convert to a list of data frames
  lapply(decoded, function(x) {
    m <- t(sapply(1:length(x), function(n) x[[n]]))
    data.frame(class_name = as.character(m[,1]),
               class_description = as.character(m[,2]),
               score = as.numeric(m[,3]),
               stringsAsFactors = FALSE)
  })
}


#' Preprocesses a tensor or array encoding a batch of images.
#'
#' @param x Input Numpy or symbolic tensor, 3D or 4D.
#' @param data_format Data format of the image tensor/array.
#' @param mode One of "caffe", "tf", or "torch"
#'   - caffe: will convert the images from RGB to BGR,
#'     then will zero-center each color channel with
#'     respect to the ImageNet dataset,
#'     without scaling.
#'   - tf: will scale pixels between -1 and 1, sample-wise.
#'   - torch: will scale pixels between 0 and 1 and then
#'     will normalize each channel with respect to the
#'     ImageNet dataset.
#'
#' @return Preprocessed tensor or array.
#'
#' @export
imagenet_preprocess_input <- function(x, data_format = NULL, mode = "caffe") {
  args <- list(
    x = x,
    # we use the vgg16 function which is the same as imagenet_utils
    preprocessor = keras$applications$vgg16$preprocess_input
  )
  if (keras_version() >= "2.0.9") {
    args$data_format <- data_format
    # no longer exists in 2.2
    if (tensorflow::tf_version() <= "2.1")
      args$mode <- mode
  }
  do.call(preprocess_input, args)
}


#' MobileNet model architecture.
#'
#' @details
#'
#' The `mobilenet_preprocess_input()` function should be used for image
#' preprocessing. To load a saved instance of a MobileNet model use
#' the `mobilenet_load_model_hdf5()` function. To prepare image input
#' for MobileNet use `mobilenet_preprocess_input()`. To decode
#' predictions use `mobilenet_decode_predictions()`.
#'
#' @inheritParams imagenet_decode_predictions
#' @inheritParams load_model_hdf5
#' @inheritParams application_xception
#'
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(224, 224, 3)` (with
#'   `channels_last` data format) or (3, 224, 224) (with `channels_first` data
#'   format). It should have exactly 3 inputs channels, and width and height
#'   should be no smaller than 32. E.g. `(200, 200, 3)` would be one valid
#'   value.
#' @param alpha controls the width of the network.
#'    - If `alpha` < 1.0, proportionally decreases the number of filters in each layer.
#'    - If `alpha` > 1.0, proportionally increases the number of filters in each layer.
#'    - If `alpha` = 1, default number of filters from the paper are used at each layer.
#' @param depth_multiplier depth multiplier for depthwise convolution (also
#'   called the resolution multiplier)
#' @param dropout dropout rate
#' @param include_top whether to include the fully-connected layer at the top of
#'   the network.
#' @param weights `NULL` (random initialization), `imagenet` (ImageNet
#'   weights), or the path to the weights file to be loaded.
#' @param input_tensor optional Keras tensor (i.e. output of `layer_input()`)
#'   to use as image input for the model.
#' @param pooling Optional pooling mode for feature extraction when
#'   `include_top` is `FALSE`.
#'     - `NULL` means that the output of the model will be the 4D tensor output
#'        of the last convolutional layer.
#'     - `avg` means that global average pooling will be applied to the output
#'        of the last convolutional layer, and thus the output of the model will
#'        be a 2D tensor.
#'     - `max` means that global max pooling will be applied.
#' @param classes optional number of classes to classify images into, only to be
#'   specified if `include_top` is TRUE, and if no `weights` argument is
#'   specified.
#' @param x input tensor, 4D
#'
#' @return `application_mobilenet()` and `mobilenet_load_model_hdf5()` return a
#'   Keras model instance. `mobilenet_preprocess_input()` returns image input
#'   suitable for feeding into a mobilenet model. `mobilenet_decode_predictions()`
#'   returns a list of data frames with variables `class_name`, `class_description`,
#'   and `score` (one data frame per sample in batch input).
#'
#' @section Reference:
#'   - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861v1.pdf).
#'
#' @export
application_mobilenet <-
function(input_shape = NULL,
         alpha = 1.0,
         depth_multiplier = 1L,
         dropout = 0.001,
         include_top = TRUE,
         weights = "imagenet",
         input_tensor = NULL,
         pooling = NULL,
         classes = 1000L,
         classifier_activation='softmax',
         ...) {
  args <- capture_args(match.call(), list(
    input_shape = normalize_shape,
    classes = as.integer,
    depth_multiplier = as.integer))
  do.call(keras$applications$MobileNet, args)
}


#' @rdname application_mobilenet
#' @export
mobilenet_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$mobilenet$preprocess_input)
}

#' @rdname application_mobilenet
#' @export
mobilenet_decode_predictions <- function(preds, top = 5) {
  imagenet_decode_predictions(preds, top)
}


#' @rdname application_mobilenet
#' @export
mobilenet_load_model_hdf5 <- function(filepath) {

  custom_objects <- list(
    relu6 = keras$applications$mobilenet$relu6
  )

  if (keras_version() < "2.1.5")
    custom_objects$DepthwiseConv2D <- keras$applications$mobilenet$DepthwiseConv2D

  load_model_hdf5(filepath, custom_objects = custom_objects)
}



#' MobileNetV2 model architecture
#'
#' @inheritParams application_mobilenet
#'
#' @return `application_mobilenet_v2()` and `mobilenet_v2_load_model_hdf5()` return a
#'   Keras model instance. `mobilenet_v2_preprocess_input()` returns image input
#'   suitable for feeding into a mobilenet v2 model. `mobilenet_v2_decode_predictions()`
#'   returns a list of data frames with variables `class_name`, `class_description`,
#'   and `score` (one data frame per sample in batch input).
#'
#' @section Reference:
#' - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
#'
#' @seealso application_mobilenet
#'
#' @export
application_mobilenet_v2 <-
function(input_shape = NULL,
         alpha = 1.0,
         include_top = TRUE,
         weights = "imagenet",
         input_tensor = NULL,
         pooling = NULL,
         classes = 1000,
         classifier_activation = 'softmax',
         ...)
{
  args <- capture_args(match.call(), list(
    input_shape = normalize_shape,
    classes = as.integer))
  do.call(keras$applications$MobileNetV2, args)
}

#' @rdname application_mobilenet_v2
#' @export
mobilenet_v2_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$mobilenetv2$preprocess_input)
}

#' @rdname application_mobilenet_v2
#' @export
mobilenet_v2_decode_predictions <- function(preds, top = 5) {
  imagenet_decode_predictions(preds, top)
}


#' @rdname application_mobilenet_v2
#' @export
mobilenet_v2_load_model_hdf5 <- function(filepath) {

  custom_objects <- list(
    relu6 = keras$applications$mobilenetv2$mobilenet_v2$relu6
  )

  if (keras_version() < "2.1.5")
    custom_objects$DepthwiseConv2D <- keras$applications$mobilenet$DepthwiseConv2D

  load_model_hdf5(filepath, custom_objects = custom_objects)
}


#' Instantiates the MobileNetV3Large architecture
#'
#' @details
#' Reference:
#' - [Searching for MobileNetV3](
#'     https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
#'
#' The following table describes the performance of MobileNets v3:
#' ------------------------------------------------------------------------
#' MACs stands for Multiply Adds
#'
#' |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
#' |---|---|---|---|---|
#' | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
#' | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
#' | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
#' | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
#' | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
#' | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' @note
#' Each Keras application typically expects a specific kind of input preprocessing.
#' For ModelNetV3, by default input preprocessing is included as a part of the
#' model (as a `Rescaling` layer), and thus
#' a preprocessing function is not necessary. In this use case, ModelNetV3 models expect their inputs
#' to be float tensors of pixels with values in the `[0-255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to FALSE.
#' With preprocessing disabled ModelNetV3 models expect their inputs to be float
#' tensors of pixels with values in the `[-1, 1]` range.
#'
#' @param input_shape Optional shape vector, to be specified if you would
#' like to use a model with an input image resolution that is not
#' `c(224, 224, 3)`.
#' It should have exactly 3 inputs channels `c(224, 224, 3)`.
#' You can also omit this option if you would like
#' to infer input_shape from an input_tensor.
#' If you choose to include both input_tensor and input_shape then
#' input_shape will be used if they match, if the shapes
#' do not match then we will throw an error.
#' E.g. `c(160, 160, 3)` would be one valid value.
#'
#' @param alpha controls the width of the network. This is known as the
#' depth multiplier in the MobileNetV3 paper, but the name is kept for
#' consistency with MobileNetV1 in Keras.
#' - If `alpha` < 1.0, proportionally decreases the number
#'     of filters in each layer.
#' - If `alpha` > 1.0, proportionally increases the number
#'     of filters in each layer.
#' - If `alpha` = 1, default number of filters from the paper
#'     are used at each layer.
#'
#' @param minimalistic In addition to large and small models this module also
#' contains so-called minimalistic models, these models have the same
#' per-layer dimensions characteristic as MobilenetV3 however, they don't
#' utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
#' and 5x5 convolutions). While these models are less efficient on CPU, they
#' are much more performant on GPU/DSP.
#'
#' @param include_top Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights String, one of `NULL` (random initialization),
#' 'imagenet' (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor Optional Keras tensor (i.e. output of
#' `layer_input()`)
#' to use as image input for the model.
#'
#' @param pooling String, optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model
#'     will be the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a
#'     2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes Integer, optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified.
#'
#' @param dropout_rate fraction of the input units to drop on the last layer.
#'
#' @param classifier_activation A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top = TRUE`. Set
#' `classifier_activation = NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing Boolean, whether to include the preprocessing
#' layer (`Rescaling`) at the bottom of the network. Defaults to `TRUE`.
#'
#' @returns A keras `Model` instance
#' @name application_mobilenet_v3
#' @rdname application_mobilenet_v3
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large>
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small>
#'   +  <https://keras.io/api/applications/>
#' @export
application_mobilenet_v3_large <-
function(input_shape = NULL,
         alpha = 1.0,
         minimalistic = FALSE,
         include_top = TRUE,
         weights = "imagenet",
         input_tensor = NULL,
         classes = 1000L,
         pooling = NULL,
         dropout_rate = 0.2,
         classifier_activation = "softmax",
         include_preprocessing = TRUE)
{
  require_tf_version("2.4", "application_mobilenet_v3_large")
  args <- capture_args(match.call(), list(
    classes = as.integer,
    input_shape = normalize_shape))
  do.call(keras$applications$MobileNetV3Large, args)
}

#' @export
#' @rdname application_mobilenet_v3
application_mobilenet_v3_small <-
function(input_shape = NULL,
         alpha = 1.0,
         minimalistic = FALSE,
         include_top = TRUE,
         weights = "imagenet",
         input_tensor = NULL,
         classes = 1000L,
         pooling = NULL,
         dropout_rate = 0.2,
         classifier_activation = "softmax",
         include_preprocessing = TRUE)
{
  require_tf_version("2.4", "application_mobilenet_v3_small")
  args <- capture_args(match.call(), list(
    classes = as.integer,
    input_shape = normalize_shape))
  do.call(keras$applications$MobileNetV3Small, args)
}

#' Instantiates the DenseNet architecture.
#'
#' @details
#'
#' Optionally loads weights pre-trained
#' on ImageNet. Note that when using TensorFlow,
#' for best performance you should set
#' `image_data_format='channels_last'` in your Keras config
#' at ~/.keras/keras.json.
#'
#' The model and the weights are compatible with
#' TensorFlow, Theano, and CNTK. The data format
#' convention used by the model is the one
#' specified in your Keras config file.
#'
#' @param blocks numbers of building blocks for the four dense layers.
#' @param include_top whether to include the fully-connected layer at the top
#'   of the network.
#' @param weights one of `NULL` (random initialization), 'imagenet'
#'   (pre-training on ImageNet), or the path to the weights file to be loaded.
#' @param input_tensor optional Keras tensor (i.e. output of `layer_input()`)
#'   to use as image input for the model.
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(224, 224, 3)`
#'   (with `channels_last` data format) or `(3, 224, 224)` (with
#'   `channels_first` data format). It should have exactly 3 inputs channels.
#' @param pooling optional pooling mode for feature extraction when
#'   `include_top` is `FALSE`.
#'      - `NULL` means that the output of the model will be the 4D tensor output
#'        of the last convolutional layer.
#'     - `avg` means that global average pooling will be applied to the output
#'        of the last convolutional layer, and thus the output of the model
#'        will be a 2D tensor.
#'     - `max` means that global max pooling will be applied.
#' @param classes optional number of classes to classify images into, only to be
#'   specified if `include_top` is TRUE, and if no `weights` argument is
#'   specified.
#' @param data_format data format of the image tensor.
#' @param x a 3D or 4D array consists of RGB values within `[0, 255]`.
#'
#' @export
application_densenet <- function(blocks, include_top = TRUE, weights = "imagenet",
                                 input_tensor = NULL, input_shape = NULL,
                                 pooling = NULL, classes = 1000) {

  keras$applications$densenet$DenseNet(
    blocks = as.integer(blocks),
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )

}

#' @rdname application_densenet
#' @export
application_densenet121 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
                                    input_shape = NULL, pooling = NULL, classes = 1000) {
  keras$applications$DenseNet121(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' @rdname application_densenet
#' @export
application_densenet169 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
                                    input_shape = NULL, pooling = NULL, classes = 1000) {
  keras$applications$DenseNet169(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' @rdname application_densenet
#' @export
application_densenet201 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL,
                                    input_shape = NULL, pooling = NULL, classes = 1000) {
  keras$applications$DenseNet201(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' @rdname application_densenet
#' @export
densenet_preprocess_input <- function(x, data_format = NULL) {
  preprocess_input(x, keras$applications$densenet$preprocess_input)
}

#' Instantiates a NASNet model.
#'
#' Note that only TensorFlow is supported for now,
#' therefore it only works with the data format
#' `image_data_format='channels_last'` in your Keras config
#' at `~/.keras/keras.json`.
#'
#' @param input_shape Optional shape list, the input shape is by default `(331, 331, 3)`
#'   for NASNetLarge and `(224, 224, 3)` for NASNetMobile It should have exactly 3
#'   inputs channels, and width and height should be no smaller than 32. E.g.
#'   `(224, 224, 3)` would be one valid value.
#' @param penultimate_filters Number of filters in the penultimate layer.
#'   NASNet models use the notation `NASNet (N @ P)`, where:
#'     - N is the number of blocks
#'     - P is the number of penultimate filters
#' @param num_blocks Number of repeated blocks of the NASNet model. NASNet
#'   models use the notation `NASNet (N @ P)`, where:
#'     - N is the number of blocks
#'     - P is the number of penultimate filters
#' @param stem_block_filters Number of filters in the initial stem block
#' @param skip_reduction Whether to skip the reduction step at the tail end
#'   of the network. Set to `FALSE` for CIFAR models.
#' @param filter_multiplier Controls the width of the network.
#'   - If `filter_multiplier` < 1.0, proportionally decreases the number of
#'     filters in each layer.
#'   - If `filter_multiplier` > 1.0, proportionally increases the number of
#'     filters in each layer. - If `filter_multiplier` = 1, default number of
#'     filters from the paper are used at each layer.
#' @param include_top Whether to include the fully-connected layer at the top
#'   of the network.
#' @param weights `NULL` (random initialization) or `imagenet` (ImageNet weights)
#' @param input_tensor Optional Keras tensor (i.e. output of `layer_input()`)
#'   to use as image input for the model.
#' @param pooling Optional pooling mode for feature extraction when
#'   `include_top` is `FALSE`.
#'     - `NULL` means that the output of the model will be the 4D tensor output
#'       of the last convolutional layer.
#'     - `avg` means that global average pooling will be applied to the output
#'       of the last convolutional layer, and thus the output of the model will
#'       be a 2D tensor.
#'     - `max` means that global max pooling will be applied.
#' @param classes Optional number of classes to classify images into, only to be
#'   specified if `include_top` is TRUE, and if no `weights` argument is
#'   specified.
#' @param default_size Specifies the default image size of the model
#' @param x a 4D array consists of RGB values within `[0, 255]`.
#'
#' @export
application_nasnet <- function(input_shape = NULL, penultimate_filters = 4032L,
                               num_blocks = 6L, stem_block_filters = 96L,
                               skip_reduction = TRUE, filter_multiplier = 2L,
                               include_top = TRUE, weights = NULL,
                               input_tensor = NULL, pooling = NULL,
                               classes = 1000, default_size = NULL) {

  keras$applications$nasnet$NASNet(
    input_shape = normalize_shape(input_shape),
    penultimate_filters = as.integer(penultimate_filters),
    num_blocks = as.integer(num_blocks),
    stem_block_filters = as.integer(stem_block_filters),
    skip_reduction = skip_reduction,
    filter_multiplier = filter_multiplier,
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    pooling = pooling,
    classes = as.integer(classes),
    default_size = default_size
  )

}

#' @rdname application_nasnet
#' @export
application_nasnetlarge <- function(input_shape = NULL, include_top = TRUE, weights = NULL,
                               input_tensor = NULL, pooling = NULL, classes = 1000) {

  keras$applications$NASNetLarge(
    input_shape = normalize_shape(input_shape),
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    pooling = pooling,
    classes = as.integer(classes)
  )

}

#' @rdname application_nasnet
#' @export
application_nasnetmobile <- function(input_shape = NULL, include_top = TRUE, weights = NULL,
                                    input_tensor = NULL, pooling = NULL, classes = 1000) {

  keras$applications$NASNetMobile(
    input_shape = normalize_shape(input_shape),
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    pooling = pooling,
    classes = as.integer(classes)
  )

}

#' Instantiates the EfficientNetB0 architecture
#'
#' @details
#' Reference:
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' EfficientNet models expect their inputs to be float tensors of pixels with values in the `[0-255]` range.
#'
#' @note
#' Each Keras Application typically expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus a calling a preprocessing function is not necessary.
#'
#' @inheritParams application_xception
#'
#' @param input_shape Optional shape list, only to be specified
#' if `include_top` is FALSE.
#' It should have exactly 3 inputs channels.
#'
#'
#' @name application_efficientnet
#' @rdname application_efficientnet
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0>
#'   +  <https://keras.io/api/applications/>
#' @export
application_efficientnet_b0 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b0")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB0, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b1 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b1")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB1, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b2 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b2")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB2, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b3 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b3")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB3, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b4 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b4")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB4, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b5 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b5")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB5, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b6 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b6")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB6, args)
}

#' @export
#' @rdname application_efficientnet
application_efficientnet_b7 <-
function(include_top = TRUE, weights = "imagenet",
         input_tensor = NULL, input_shape = NULL,
         pooling = NULL, classes = 1000L,
         classifier_activation = "softmax",
         ...)
{
    require_tf_version("2.3", "application_efficientnet_b7")
    args <- capture_args(match.call(), list(classes = as.integer, input_shape = normalize_shape))
    do.call(keras$applications$EfficientNetB7, args)
}


#' @rdname application_nasnet
#' @export
nasnet_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$nasnet$preprocess_input)
}

preprocess_input <- function(x, preprocessor, ...) {
  preprocessor(keras_array(x), ...)
}

verify_application_prerequistes <- function() {

  if (!have_h5py())
    stop("The h5py Python package is required to use pre-built Keras models", call. = FALSE)

}
