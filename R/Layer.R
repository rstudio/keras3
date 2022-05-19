#' (Deprecated) Create a custom Layer
#'
#' This function is maintained but deprecated. Please use `new_layer_class()` or
#' `%py_class%` to define custom layers.
#'
#' @param classname the name of the custom Layer.
#' @param initialize a function. This is where you define the arguments used to further
#'  build your layer. For example, a dense layer would take the `units` argument.
#'  You should always call \code{super()$`__init__()`} to initialize the base
#'  inherited layer.
#' @param build a function that takes `input_shape` as argument. This is where you will
#'  define your weights. Note that if your layer doesn't define trainable weights then
#'  you need not implement this method.
#' @param call This is where the layer's logic lives. Unless you want your layer to
#'  support masking, you only have to care about the first argument passed to `call`
#'  (the input tensor).
#' @param compute_output_shape a function that takes `input_shape` as an argument. In
#'  case your layer modifies the shape of its input, you should specify here the
#'  shape transformation logic. This allows Keras to do automatic shape inference.
#'  If you don't modify the shape of the input then you need not implement this
#'  method.
#' @param ... Any other methods and/or attributes can be specified using named
#'  arguments. They will be added to the layer class.
#' @param inherit the Keras layer to inherit from.
#' @return A function that wraps `create_layer`, similar to `keras::layer_dense`.
#' @examples
#' \dontrun{
#'
#' layer_dense2 <- Layer(
#'   "Dense2",
#'
#'   initialize = function(units) {
#'     super()$`__init__`()
#'     self$units <- as.integer(units)
#'   },
#'
#'   build = function(input_shape) {
#'     print(class(input_shape))
#'     self$kernel <- self$add_weight(
#'       name = "kernel",
#'       shape = list(input_shape[[2]], self$units),
#'       initializer = "uniform",
#'       trainable = TRUE
#'     )
#'   },
#'
#'   call = function(x) {
#'     tensorflow::tf$matmul(x, self$kernel)
#'   },
#'
#'   compute_output_shape = function(input_shape) {
#'     list(input_shape[[1]], self$units)
#'   }
#'
#' )
#'
#' l <- layer_dense2(units = 10)
#' l(matrix(runif(10), ncol = 1))
#'
#' }
#'
#' @keywords internal
#' @export
Layer <-
function(classname, initialize, build = NULL, call = NULL,
         compute_output_shape = NULL, ...,
         inherit = keras::keras$layers$Layer) {

  public <- capture_args(match.call(), ignore = c("classname", "inherit"))
  for(ignore_if_null in c("build", "call", "compute_output_shape"))
    public[[ignore_if_null]] <- public[[ignore_if_null]]

  inherit <- substitute(inherit)
  parent_env <- parent.frame()

  # R6Class() calls substitute() on inherit;
  r_cls <- eval(as.call(list(
    quote(R6::R6Class),
    classname = classname,
    public = public,
    active = NULL,
    inherit = inherit,
    cloneable = FALSE,
    parent_env = parent_env
  )))

  create_layer_wrapper(r_cls)
}
