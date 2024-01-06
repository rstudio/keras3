

# TODO: still need to add tests for all these.
# TODO: should all optimizer accept a plain R function to `learning_rate`?


#' Create a new learning rate schedule type
#'
#' @param classname string
#' @param ... methods and properties of the schedule class
#' @param call function which takes a step argument (scalar integer tensor, the
#'   current training step count, and returns the new learning rate). For
#'   tracking additional state, objects `self` and `private` are automatically
#'   injected into the scope of the function.
#' @param initialize,get_config Additional recommended methods to implement.
#'
#'  +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule>
#' @return A `LearningRateSchedule` class generator.
#' @export
new_learning_rate_schedule_class <-
function(classname, ..., initialize = NULL, call, get_config = NULL) {
  members <- capture_args2(ignore = "classname")
  members <- drop_nulls(members)
  members <- rename_to_dunder(members, "call")
  if (!is.null(members[["call"]])) {
    if ("__call__" %in% names(members))
      warning("`call()` method is ignored, superceded by `__call__`() method.")
    else
      names(members)[match("call", names(members))] <- "__call__"
  }

  new_py_class(
    classname,
    members = members,
    inherit = keras$optimizers$schedules$LearningRateSchedule,
    parent_env = parent.frame(),
    convert = TRUE
  )
}


rename_to_dunder <- function(members, nms) {
  if(anyDuplicated(names(members)))
    stop("All names must be unique")
  for (nm in nms) {
    .__nm__ <- paste0("__", nm, "__")
    if (nm %in% names(members)) {
      if (.__nm__ %in% names(members))
        warning("`", nm, "` method is ignored, superceded by `", .__nm__, "` method.")
      else
        names(members)[match(nm, names(members))] <- .__nm__
    }
  }
  members
}
