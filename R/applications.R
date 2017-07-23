

#' Xception V1 model for Keras.
#' 
#' @details 
#' On ImageNet, this model gets to a top-1 validation accuracy of 0.790
#' and a top-5 validation accuracy of 0.945.
#' 
#' Do note that the input image format for this model is different than for
#' the VGG16 and ResNet models (299x299 instead of 224x224). 
#' 
#' The `xception_preprocess_input()` function should be used for image 
#' preprocessing.
#' 
#' This application is only available when using the TensorFlow back-end.
#' 
#' @param x Input tensor for preprocessing
#' @param include_top whether to include the fully-connected layer at the top of
#'   the network.
#' @param weights one of `NULL` (random initialization) or "imagenet" 
#'   (pre-training on ImageNet).
#' @param input_tensor optional Keras tensor to use as image input for the
#'   model.
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(299, 299, 3)`. It should 
#'   have exactly 3 inputs channels, and width and height should be no smaller 
#'   than 71. E.g. `(150, 150, 3)` would be one valid value.
#' @param pooling Optional pooling mode for feature extraction when 
#'   `include_top` is `FALSE`. 
#'   - `NULL` means that the output of the model will be the 4D tensor output 
#'      of the last convolutional layer. 
#'   - `avg` means that global average pooling will be applied to the output of
#'      the last convolutional layer, and thus the output of the model will be
#'      a 2D tensor. 
#'   - `max` means that global max pooling will be applied.
#' @param classes optional number of classes to classify images into, only to be
#'   specified if `include_top` is TRUE, and if no `weights` argument is 
#'   specified.
#'   
#' @section Reference: 
#'   - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)   
#'   
#' @return A Keras model instance.
#'   
#' @export
application_xception <- function(include_top = TRUE, weights = "imagenet", 
                                 input_tensor = NULL, input_shape = NULL, 
                                 pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$Xception(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = input_shape,
    pooling = pooling,
    classes = as.integer(classes)
  )
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
#'   than 48. E.g. `(200, 200, 3)` would be one valid value.
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
#' dim(x) <- c(1, dim(x))
#' x <- imagenet_preprocess_input(x)
#' 
#' features <- model %>% predict(x)
#' }
#' @export
application_vgg16 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, 
                              pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$VGG16(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = input_shape,
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' @rdname application_vgg   
#' @export
application_vgg19 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, 
                              pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$VGG19(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = input_shape,
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' ResNet50 model for Keras.
#' 
#' @details Optionally loads weights pre-trained on ImageNet. 
#' 
#' The `imagenet_preprocess_input()` function should be used for image 
#' preprocessing.
#'   
#' @inheritParams application_xception
#'   
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(224, 224, 3)`. It should
#'   have exactly 3 inputs channels, and width and height should be no smaller
#'   than 197. E.g. `(200, 200, 3)` would be one valid value.
#'   
#' @return A Keras model instance.
#'   
#' @section Reference: - [Deep Residual Learning for Image 
#'   Recognition](https://arxiv.org/abs/1512.03385)
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
#' dim(x) <- c(1, dim(x))
#' x <- imagenet_preprocess_input(x)
#' 
#' # make predictions then decode and print them
#' preds <- model %>% predict(x)
#' imagenet_decode_predictions(preds, top = 3)[[1]]
#' }   
#' @export
application_resnet50 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL, 
                                 pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$ResNet50(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = input_shape,
    pooling = pooling,
    classes = as.integer(classes)
  )
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
#'  - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
#' 
#' @export
application_inception_v3 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
                                     pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$InceptionV3(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = input_shape,
    pooling = pooling,
    classes = as.integer(classes)
  )
}

#' @rdname application_inception_v3
#' @export
inception_v3_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$inception_v3$preprocess_input)
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
  decoded <- keras$applications$imagenet_utils$decode_predictions(
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


#' Preprocesses a tensor encoding a batch of images.
#' 
#' @param x input tensor, 4D
#' 
#' @return Preprocessed tensor
#' 
#' @export
imagenet_preprocess_input <- function(x) {
  preprocess_input(x, keras$applications$imagenet_utils$preprocess_input)
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
#' MobileNet is currently only supported with the TensorFlow backend.
#'
#' @inheritParams imagenet_decode_predictions
#' @inheritParams load_model_hdf5
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
#' @param weights `NULL` (random initialization) or `imagenet` (ImageNet
#'   weights)
#' @param input_tensor optional Keras tensor (i.e. output of `layers.Input()`)
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
application_mobilenet <- function(input_shape = NULL, alpha = 1.0, depth_multiplier = 1, dropout = 0.001, 
                                  include_top = TRUE, weights = "imagenet", input_tensor = NULL, pooling = NULL, 
                                  classes = 1000) {
  keras$applications$MobileNet(
    input_shape = as_integer_tuple(input_shape),
    alpha = alpha,
    depth_multiplier = as.integer(depth_multiplier),
    dropout = dropout,
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    pooling = pooling,
    classes = as.integer(classes)
  )
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
  load_model_hdf5(filepath, custom_objects = list(
    relu6 = keras$applications$mobilenet$relu6,
    DepthwiseConv2D = keras$applications$mobilenet$DepthwiseConv2D
  ))
}



# the preprocesssing functions modify the ndarray in place
# so we can't pass an R marshalled array (since it points to
# R managed memory numpy won't allow writing to it). this 
# function wraps preprocessing by making a copy of the R
# array before passing it to numpy
preprocess_input <- function(x, preprocessor) {
  np <- import("numpy", convert = FALSE)
  x_np <- np$copy(x)
  preprocessor(x_np)
  py_to_r(x_np)
}


verify_application_prerequistes <- function() {

  if (!have_h5py())
    stop("The h5py Python package is required to use pre-built Keras models", call. = FALSE)
  
}

