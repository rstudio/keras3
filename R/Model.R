#' Subclass the base Keras `Model` Class
#'
#' @description
#'
#' This is for advanced use cases where you need to subclass the base Model
#' type, e.g., you want to override the `train_step()` method.
#'
#' If you just want to create or define a keras model, prefer [`keras_model()`]
#' or [`keras_model_sequential()`].
#'
#' If you just want to encapsulate some custom logic and state, and don't need
#' to customize training behavior (besides calling `self$add_loss()` in the
#' `call()` method), prefer [`Layer()`].
#'
#' @details
#' @inheritSection Layer Symbols in scope
#'
#' @param
#' initialize,update_state,result,call,train_step,predict_step,test_step,compute_loss,compute_metrics
#' Optional methods that can be overridden.
#' @inheritParams Layer
#'
#' @return A model constructor function, which you can call to create an
#'   instance of the new model type.
#' @seealso [active_property()] (e.g., for a `metrics` property implemented as a
#'   function).
#' @export
Model <-
function(classname,
         initialize = NULL,
         call = NULL,
         train_step = NULL,
         predict_step = NULL,
         test_step = NULL,
         compute_loss = NULL,
         compute_metrics = NULL,
         ...,
         public = list(),
         private = list(),
         inherit = NULL,
         parent_env = parent.frame())
{
  members <- drop_nulls(named_list(initialize, call,
                                   train_step, predict_step, test_step,
                                   compute_loss, compute_metrics))
  members <- modifyList(members, list2(...), keep.null = TRUE)
  members <- modifyList(members, public, keep.null = TRUE)

  members <- modify_intersection(members, list(
    from_config = function(x) decorate_method(x, "classmethod")
  ))

  inherit <- substitute(inherit) %||%
    quote(base::asNamespace("keras3")$keras$Model)

  new_wrapped_py_class(
    classname = classname,
    members = members,
    inherit = inherit,
    parent_env = parent_env,
    private = private
  )

}

