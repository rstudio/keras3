

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
#' @param weights `NULL` (random initialization), `imagenet` (ImageNet
#'   weights), or the path to the weights file to be loaded.
#' @param input_tensor optional Keras tensor to use as image input for the
#'   model.
#' @param input_shape optional shape list, only to be specified if `include_top`
#'   is FALSE (otherwise the input shape has to be `(299, 299, 3)`. It should
#'   have exactly 3 inputs channels, and width and height should be no smaller
#'   than 75. E.g. `(150, 150, 3)` would be one valid value.
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
    input_shape = normalize_shape(input_shape),
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
application_vgg16 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
                              pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$VGG16(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
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
    input_shape = normalize_shape(input_shape),
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
#'   than 32. E.g. `(200, 200, 3)` would be one valid value.
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
#' x <- array_reshape(x, c(1, dim(x)))
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
    input_shape = normalize_shape(input_shape),
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
#'  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
#'
#' @export
application_inception_v3 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
                                     pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$InceptionV3(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )
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
application_inception_resnet_v2 <- function(include_top = TRUE, weights = "imagenet", input_tensor = NULL, input_shape = NULL,
                                            pooling = NULL, classes = 1000) {
  verify_application_prerequistes()
  keras$applications$InceptionResNetV2(
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    input_shape = normalize_shape(input_shape),
    pooling = pooling,
    classes = as.integer(classes)
  )
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
application_mobilenet <- function(input_shape = NULL, alpha = 1.0, depth_multiplier = 1, dropout = 0.001,
                                  include_top = TRUE, weights = "imagenet", input_tensor = NULL, pooling = NULL,
                                  classes = 1000) {
  keras$applications$MobileNet(
    input_shape = normalize_shape(input_shape),
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
application_mobilenet_v2 <- function(input_shape = NULL, alpha = 1.0,  include_top = TRUE,
                                     weights = "imagenet", input_tensor = NULL, pooling = NULL, classes = 1000) {

  keras$applications$MobileNetV2(
    input_shape = normalize_shape(input_shape),
    alpha = alpha,
    include_top = include_top,
    weights = weights,
    input_tensor = input_tensor,
    pooling = pooling,
    classes = as.integer(classes)
  )

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
