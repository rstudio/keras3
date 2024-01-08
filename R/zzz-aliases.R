

.alias_roxygen <- function(backcompat_name, name) {
  strsplit(glue::glue(r"---(
    {name}

    `{backcompat_name}()` is an alias for [`{name}()`].
    See [`?{name}()`] for the full documentation.

    @inheritParams {name}
    @keywords internal
    @export

    )---"), "\n", fixed = TRUE)[[1L]]
}


#' @eval .alias_roxygen("new_callback_class", "Callback")
new_callback_class <- Callback


#' @eval .alias_roxygen("new_layer_class", "Layer")
new_layer_class <- Layer

#' @eval .alias_roxygen("new_loss_class", "Loss")
new_loss_class <- Loss

r"---(

# @eval .alias_roxygen("new_metric_class", "Metric")
new_metric_class <- Metric

# @eval .alias_roxygen("new_learning_rate_schedule_class", "LearningRateSchedule")
new_learning_rate_schedule_class <- LearningRateSchedule

# @eval .alias_roxygen("new_layer_class", "Model")
new_model_class <- Model

)---"


#' @eval .alias_roxygen("mark_active", "active_property")
mark_active <- active_property
