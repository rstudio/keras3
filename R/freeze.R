

#' Freeze and unfreeze weights
#'
#' Freeze weights in a model or layer so that they are no longer trainable.
#'
#' @param object Keras model or layer object
#' @param from Layer instance, layer name, or layer index within model
#' @param to Layer instance, layer name, or layer index within model
#' @param which layer names, integer positions, layers, logical vector (of
#'   `length(object$layers)`), or a function returning a logical vector.
#'
#' @note The `from` and `to` layer arguments are both inclusive.
#'
#'   When applied to a model, the freeze or unfreeze is a global operation over
#'   all layers in the model (i.e. layers not within the specified range will be
#'   set to the opposite value, e.g. unfrozen for a call to freeze).
#'
#'   Models must be compiled again after weights are frozen or unfrozen.
#'
#' @examples \dontrun{
# instantiate a VGG16 model
#' conv_base <- application_vgg16(
#'   weights = "imagenet",
#'   include_top = FALSE,
#'   input_shape = c(150, 150, 3)
#' )
#'
#' # freeze it's weights
#' freeze_weights(conv_base)
#'
#' conv_base
#'
#' # create a composite model that includes the base + more layers
#' model <- keras_model_sequential() %>%
#'   conv_base() %>%
#'   layer_flatten() %>%
#'   layer_dense(units = 256, activation = "relu") %>%
#'   layer_dense(units = 1, activation = "sigmoid")
#'
#' # compile
#' model %>% compile(
#'   loss = "binary_crossentropy",
#'   optimizer = optimizer_rmsprop(lr = 2e-5),
#'   metrics = c("accuracy")
#' )
#'
#' model
#' print(model, expand_nested = TRUE)
#'
#'
#'
#' # unfreeze weights from "block5_conv1" on
#' unfreeze_weights(conv_base, from = "block5_conv1")
#'
#' # compile again since we froze or unfroze weights
#' model %>% compile(
#'   loss = "binary_crossentropy",
#'   optimizer = optimizer_rmsprop(lr = 2e-5),
#'   metrics = c("accuracy")
#' )
#'
#' conv_base
#' print(model, expand_nested = TRUE)
#'
#' # freeze only the last 5 layers
#' freeze_weights(conv_base, from = -5)
#' conv_base
#' # equivalently, also freeze only the last 5 layers
#' unfreeze_weights(conv_base, to = -6)
#' conv_base
#'
#' # Freeze only layers of a certain type, e.g, BatchNorm layers
#' batch_norm_layer_class_name <- class(layer_batch_normalization())[1]
#' is_batch_norm_layer <- function(x) inherits(x, batch_norm_layer_class_name)
#'
#' model <- application_efficientnet_b0()
#' freeze_weights(model, which = is_batch_norm_layer)
#' model
#' # equivalent to:
#' for(layer in model$layers) {
#'   if(is_batch_norm_layer(layer))
#'     layer$trainable <- FALSE
#'   else
#'     layer$trainable <- TRUE
#' }
#' }
#' @export
freeze_weights <- function(object, from = NULL, to = NULL, which = NULL) {

  if (!is.null(which)) {
    if(!is.null(from) && !is.null(to))
      stop("both `which` and `from`/`to` can not be supplied")
    return(apply_which_trainable(object, which, FALSE))
  }

  # check for from and to and apply accordingly
  if (missing(from) && missing(to)) {
    object$trainable <- FALSE
  } else {
    object$trainable <- TRUE
    apply_trainable(object, from, to, FALSE)
  }

  # return model invisibly (for chaining)
  invisible(object)
}


#' @rdname freeze_weights
#' @export
unfreeze_weights <- function(object, from = NULL, to = NULL, which = NULL) {

  # object always trainable after unfreeze
  object$trainable <- TRUE

  if (!is.null(which)) {
    if(!is.null(from) && !is.null(to))
      stop("both `which` and `from`/`to` can not be supplied")
    return(apply_which_trainable(object, which, TRUE))
  }

  # apply to individual layers if requested
  if (!missing(from) || !missing(to))
    apply_trainable(object, from, to, TRUE)

  # return model invisibly (for chaining)
  invisible(object)
}

apply_trainable <- function(object, from, to, trainable) {

  # first resolve from and to into layer names
  layers <- object$layers

  # NULL means beginning and end respectively
  if (is.null(from))
    from <- layers[[1]]$name
  if (is.null(to))
    to <- layers[[length(layers)]]$name

  # layer instances become layer names
  if (is_layer(from))
    from <- from$name
  if (is_layer(to))
    to <- to$name

  # layer indexes become layer names
  if (is.numeric(from)) {
    if(from < 0)
      from <- length(layers) - abs(from) + 1
    from <- layers[[from]]$name
  }
  if (is.numeric(to)) {
    if(to < 0)
      to <- length(layers) - abs(to) + 1
    to <- layers[[to]]$name
  }

  # apply trainable property
  set_trainable <- FALSE
  for (layer in layers) {

    # flag to begin applying property
    if (layer$name == from)
      set_trainable <- TRUE

    # apply property
    if (set_trainable)
      layer$trainable <- trainable
    else
      layer$trainable <- !trainable

    # flag to stop applying property
    if (layer$name == to)
      set_trainable <- FALSE
  }
}

apply_which_trainable <- function(object, which, trainable) {

  # presumably, since user is being selective, some parts of the object are
  # still trainable
  object$trainable <- TRUE

  layers <- object$layers
  names(layers) <- vapply(layers, function(l) l$name, "", USE.NAMES = FALSE)

  if(inherits(which, "formula"))
    which <- rlang::as_function(which)

  if(is.function(which))
    which <- vapply(layers, which, TRUE, USE.NAMES = FALSE)

  if(is.logical(which))
    which <- base::which(which)

  # invert all the layers, then set just the flag for the requested layers
  for(l in layers)
    l$trainable <- !trainable

  for (i in which) {

    if (is.character(i)) {

      layer <- layers[[i]]

    } else if (is.numeric(i)) {

      if (i < 0)
        i <- length(layers) + i + 1
      layer <- layers[[i]]

    } else if (is_layer(i)) {

      layer <- i

    } else {

      stop(
        "`which` must be:layer names, index position, layers, ",
        "logical vector, or a function returning a logical vector"
      )

    }

    layer$trainable <- trainable
  }

  return(invisible(object))
}
