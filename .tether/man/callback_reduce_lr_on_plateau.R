#' Reduce learning rate when a metric has stopped improving.
#'
#' @description
#' Models often benefit from reducing the learning rate by a factor
#' of 2-10 once learning stagnates. This callback monitors a
#' quantity and if no improvement is seen for a 'patience' number
#' of epochs, the learning rate is reduced.
#'
#' # Examples
#' ```python
#' reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#'                               patience=5, min_lr=0.001)
#' model.fit(x_train, y_train, callbacks=[reduce_lr])
#' ```
#'
#' @param monitor
#' String. Quantity to be monitored.
#'
#' @param factor
#' Float. Factor by which the learning rate will be reduced.
#' `new_lr = lr * factor`.
#'
#' @param patience
#' Integer. Number of epochs with no improvement after which
#' learning rate will be reduced.
#'
#' @param verbose
#' Integer. 0: quiet, 1: update messages.
#'
#' @param mode
#' String. One of `{'auto', 'min', 'max'}`. In `'min'` mode,
#' the learning rate will be reduced when the
#' quantity monitored has stopped decreasing; in `'max'` mode it will
#' be reduced when the quantity monitored has stopped increasing; in
#' `'auto'` mode, the direction is automatically inferred from the name
#' of the monitored quantity.
#'
#' @param min_delta
#' Float. Threshold for measuring the new optimum, to only focus
#' on significant changes.
#'
#' @param cooldown
#' Integer. Number of epochs to wait before resuming normal
#' operation after the learning rate has been reduced.
#'
#' @param min_lr
#' Float. Lower bound on the learning rate.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family callbacks
#' @seealso
#' + <https:/keras.io/api/callbacks/reduce_lr_on_plateau#reducelronplateau-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau>
callback_reduce_lr_on_plateau <-
function (monitor = "val_loss", factor = 0.1, patience = 10L,
    verbose = 0L, mode = "auto", min_delta = 1e-04, cooldown = 0L,
    min_lr = 0L, ...)
{
}
