

#' (Deprecated) Base R6 class for Keras constraints
#'
#' New custom constraints are encouraged to subclass `keras$constraints$Constraint` directly.
#'
#' @docType class
#'
#' @format An [R6Class] generator object
#'
#' @section Methods:
#' \describe{
#'  \item{\code{call(w)}}{Constrain the specified weights.}
#' }
#'
#' @details You can implement a custom constraint either by creating an
#'  R function that accepts a weights (`w`) parameter, or by creating
#'  an R6 class that derives from `KerasConstraint` and implements a
#'  `call` method.
#'
#' @note
#' Models which use custom constraints cannot be serialized using
#' [save_model_hdf5()]. Rather, the weights of the model should be saved
#' and restored using [save_model_weights_hdf5()].
#'
#' @examples \dontrun{
#' CustomNonNegConstraint <- R6::R6Class(
#'   "CustomNonNegConstraint",
#'   inherit = KerasConstraint,
#'   public = list(
#'     call = function(x) {
#'        w * k_cast(k_greater_equal(w, 0), k_floatx())
#'     }
#'   )
#' )
#'
#' layer_dense(units = 32, input_shape = c(784),
#'             kernel_constraint = CustomNonNegConstraint$new())
#' }
#'
#' @seealso [constraint_unitnorm] and related constraints
#'
#' @keywords internal
#' @export
KerasConstraint <- R6::R6Class("KerasConstraint",
  public = list(
    call = function(w) {
      stop("Keras custom constraints must implement the call function")
    },
    get_config = function() {
      reticulate::dict()
    })
)

as_constraint <- function(constraint) {

  # helper to create constraint
  create_constraint <- function(call, get_config = NULL) {
    if (is.null(get_config))
      get_config <- function() dict()
    python_path <- system.file("python", package = "keras")
    tools <- import_from_path("kerastools", path = python_path)
    tools$constraint$RConstraint(call, get_config)
  }

  if (inherits(constraint, "keras.constraints.Constraint")) {
    constraint
  } else if (is.function(constraint)) {
    create_constraint(constraint)
  } else if (inherits(constraint, "KerasConstraint")) {
    create_constraint(constraint$call, constraint$get_config)
  } else {
    constraint
  }
}


# TODO: generate @family constraints, not @family constraint
