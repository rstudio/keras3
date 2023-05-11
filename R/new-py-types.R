

#' @rdname new-classes
#' @export
new_metric_class <-
function(classname, ..., initialize, update_state, result) {
  members <- capture_args(match.call(), ignore = "classname")
  new_py_class(classname, members,
              inherit = keras::keras$metrics$Metric,
              parent_env = parent.frame())
}

#' @rdname new-classes
#' @export
new_loss_class <-
function(classname, ..., call = NULL) {
  members <- capture_args(match.call(), ignore = "classname")
  members$call <- call
  new_py_class(classname, members,
              inherit = keras::keras$losses$Loss,
              parent_env = parent.frame())
}

#' @rdname new-classes
#' @export
new_callback_class <-
function(classname,
         ...,
         on_epoch_begin = NULL,
         on_epoch_end = NULL,
         on_train_begin = NULL,
         on_train_end = NULL,
         on_batch_begin = NULL,
         on_batch_end = NULL,
         on_predict_batch_begin = NULL,
         on_predict_batch_end = NULL,
         on_predict_begin = NULL,
         on_predict_end = NULL,
         on_test_batch_begin = NULL,
         on_test_batch_end = NULL,
         on_test_begin = NULL,
         on_test_end = NULL,
         on_train_batch_begin = NULL,
         on_train_batch_end = NULL) {

  members <- capture_args(match.call(), ignore = "classname")
  members <- drop_nulls(members,
    names(which(vapply(formals(sys.function()), is.null, TRUE))))

  new_py_class(classname, members,
              inherit = keras::keras$callbacks$Callback,
              parent_env = parent.frame())
}


#' @rdname new-classes
#' @export
new_model_class <-
function(classname, ...,
         initialize = NULL, call = NULL,
         train_step = NULL, predict_step = NULL, test_step = NULL,
         compute_loss = NULL, compute_metrics = NULL) {
  members <- capture_args(match.call(), ignore = "classname")
  members <- drop_nulls(members,
    names(which(vapply(formals(sys.function()), is.null, TRUE))))

  new_py_class(classname, members,
              inherit = keras::keras$Model,
              parent_env = parent.frame())
}



#' Define new keras types
#'
#' These functions can be used to make custom objects that fit in the family of
#' existing keras types. For example, `new_layer_class()` will return a class
#' constructor, an object that behaves like other layer functions such as
#' `layer_dense()`. `new_callback_class()` will return an object that behaves
#' similarly to other callback functions, like
#' `callback_reduce_lr_on_plateau()`, and so on. All arguments with a default
#' `NULL` value are optional methods that can be provided.
#'
#' `mark_active()` is a decorator that can be used to indicate functions that
#' should become active properties of the class instances.
#'
#' @rdname new-classes
#' @param classname The classname as a string. Convention is for the classname
#'   to be a CamelCase version of the constructor.
#' @param ... Additional fields and methods for the new type.
#' @param initialize,build,call,get_config,on_epoch_begin,on_epoch_end,on_train_begin,on_train_end,on_batch_begin,on_batch_end,on_predict_batch_begin,on_predict_batch_end,on_predict_begin,on_predict_end,on_test_batch_begin,on_test_batch_end,on_test_begin,on_test_end,on_train_batch_begin,on_train_batch_end,update_state,result,train_step,predict_step,test_step,compute_loss,compute_metrics Optional methods that can be overridden.
#' @param x A function that should be converted to an active property of the class type.
#'
#' @return A new class generator object that inherits from the appropriate Keras
#'   base class.
#' @export
new_layer_class <-
function(classname, ...,
         initialize = NULL, build = NULL, call = NULL, get_config = NULL) {
  members <- capture_args(match.call(),  ignore = "classname")
  members <- drop_nulls(members,
    names(which(vapply(formals(sys.function()), is.null, TRUE))))

  type <- new_py_class(classname, members,
                      inherit = keras$layers$Layer,
                      parent_env = parent.frame())

  create_layer_wrapper(type)
}
