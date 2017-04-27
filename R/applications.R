

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
  keras$applications$xception$preprocess_input(x)
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
  keras$applications$inception_v3$preprocess_input(x)
}

#' Decodes the prediction of an ImageNet model.
#' 
#' @param preds Tensor encoding a batch of predictions.
#' @param top integer, how many top-guesses to return.
#'   
#' @return A list of lists of top class prediction lists `(class_name,
#'   class_description, score)`. One list of lists per sample in batch input.
#'   
#' @export
imagenet_decode_predictions <- function(preds, top = 5) {
  keras$applications$imagenet_utils$decode_predictions(
    preds = preds,
    top = as.integer(top)
  )
}

#' Preprocesses a tensor encoding a batch of images.
#' 
#' @param x input tensor, 4D
#' 
#' @return Preprocessed tensor
#' 
#' @export
imagenet_preprocess_input <- function(x) {
  keras$applications$imagenet_utils$preprocess_input(x)
}

verify_application_prerequistes <- function() {
  if (!have_h5py())
    stop("The h5py Python package is required to use pre-built Keras models", call. = FALSE)
  
}

