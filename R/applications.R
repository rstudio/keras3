
#' @rdname application_xception
#' @export
xception_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$xception$preprocess_input)
}

#  @inheritParams application_efficientnet
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
resnet_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$resnet$preprocess_input)
}

#' @export
#' @rdname application_resnet
resnet_v2_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$resnet_v2$preprocess_input)
}





#' @rdname application_inception_v3
#' @export
inception_v3_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$inception_v3$preprocess_input)
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


#' Application helpers
#'
#' @rdname application_densenet
#' @export
densenet_preprocess_input <- function(x, data_format = NULL) {
  preprocess_input(x, keras$applications$densenet$preprocess_input)
}

#' Application helpers
#'
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
