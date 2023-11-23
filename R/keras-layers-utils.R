

#' Torch module wrapper layer.
#'
#' @description
#' `layer_torch_module_wrapper` is a wrapper class that can turn any
#' `torch.nn.Module` into a Keras layer, in particular by making its
#' parameters trackable by Keras.
#'
#' # Examples
#' Here's an example of how the `layer_torch_module_wrapper` can be used with vanilla
#' PyTorch modules.
#'
#' ```{r, eval = FALSE}
#' torch <- import("torch")
#' nn <- import("torch.nn")
#' nnf <- import("torch.nn.functional")
#'
#' Classifier(keras$Model) %py_class% {
#'
#'   initialize <- function(...) {
#'     super$initialize(...)
#'
#'     self$conv1 <- layer_torch_module_wrapper(module = nn$Conv2d(
#'       in_channels=1L, out_channels=32L, kernel_size=c(3L, 3L)
#'     ))
#'     self$conv2 <- layer_torch_module_wrapper(module = nn$Conv2d(
#'       in_channels=32L, out_channels=64L, kernel_size=c(3L, 3L)
#'     ))
#'     self$pool <- nn$MaxPool2d(kernel_size=c(2L, 2L))
#'     self$flatten <- nn$Flatten()
#'     self$dropout <- nn$Dropout(p=0.5)
#'     self$fc <- layer_torch_module_wrapper(module = nn$Linear(
#'       1600L, 10L
#'     ))
#'   }
#'
#'   call <- function(inputs) {
#'     x <- nnf$relu(self$conv1(inputs))
#'     x <- self$pool(x)
#'     x <- nnf$relu(self$conv2(x))
#'     x <- self$pool(x)
#'     x <- self$flatten(x)
#'     x <- self$dropout(x)
#'     x <- self$fc(x)
#'     nnf$softmax(x, dim=1L)
#'   }
#'
#' }
#'
#' model <- Classifier()
#' model$build(shape(1, 28, 28))
#' print("Output shape:", model(torch$ones(shape(1L, 1L, 28L, 28L))))
#'
#' model %>% compile(
#'     loss="sparse_categorical_crossentropy",
#'     optimizer="adam",
#'     metrics="accuracy"
#' )
#' model %>% fit(train_loader, epochs=5)
#' ```
#'
#' @param module
#' `torch.nn.Module` instance. If it's a `LazyModule`
#' instance, then its parameters must be initialized before
#' passing the instance to `layer_torch_module_wrapper` (e.g. by calling
#' it once).
#'
#' @param name
#' The name of the layer (string).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family utils
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TorchModuleWrapper>
#' @tether keras.layers.TorchModuleWrapper
layer_torch_module_wrapper <-
function (object, module, name = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$TorchModuleWrapper, object, args)
}
