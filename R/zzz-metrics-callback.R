

callback_view_metrics <- Callback(
  "ViewMetricsCallback",

  public = list(

    initialize = function(view_metrics = FALSE, initial_epoch = 1) {
      private$view_metrics <- view_metrics
      private$initial_epoch <- initial_epoch
    },

    on_train_begin = function(logs = NULL) {
      if (tfruns::is_run_active()) {
        private$write_params(self$params)
        private$write_model_info(self$model)
      }
    },

    on_epoch_end = function(epoch, logs = NULL) {

      if ((epoch - private$initial_epoch) == 0) {
        # first epoch

        # logs is a dict/named list
        private$metrics <- lapply(logs, function(x) logical())

        sleep <- 0.25 # 0.5

      } else {

        sleep <- 0.1

      }

      # handle metrics
      private$on_metrics(logs, sleep)

    }
  ),

  private = list(
    on_metrics = function(logs, sleep) {

      # record metrics
      metrics <- private$metrics
      for (metric in names(metrics)) {
        # guard against metrics not yet available by using NA
        # when a named metrics isn't passed in 'logs'
        append(metrics[[metric]]) <- mean(logs[[metric]] %||% NA)
      }
      private$metrics <- metrics

      # create history object and convert to metrics data frame

      history <- keras_training_history(self$params, private$metrics)
      metrics <- private$as_metrics_df(history)

      # view metrics if requested
      if (private$view_metrics) {

        # create the metrics_viewer or update if we already have one
        metrics_viewer <- private$metrics_viewer
        if (is.null(metrics_viewer)) {
          private$metrics_viewer <- tfruns::view_run_metrics(metrics)
        } else {
          tfruns::update_run_metrics(metrics_viewer, metrics)
        }

        # pump events
        utils::process.events()
        Sys.sleep(sleep)
        utils::process.events()
      }
      # record metrics
      tfruns::write_run_metadata("metrics", metrics)
    },

    # convert keras history to metrics data frame suitable for plotting
    as_metrics_df = function(history) {

      # create metrics data frame
      metrics <- lapply(history$metrics, function(m) sapply(m, as.numeric))
      df <- as.data.frame(metrics)

      # pad to epochs if necessary
      pad <- history$params$epochs - nrow(df)
      pad_data <- list()

      metric_names <- names(history$metrics)

      for (metric in metric_names)
        pad_data[[metric]] <- rep_len(NA, pad)

      df <- rbind(df, pad_data)

      # return df
      df
    },

    write_params = function(params) {
      properties <- list(
        samples            = params$samples,
        validation_samples = params$validation_samples,
        epochs             = params$epochs,
        batch_size         = params$batch_size
      )
      tfruns::write_run_metadata("properties", properties)
    },

    write_model_info = function(model) {
      tryCatch({
        model_info <- list()
        model_info$model <- py_str(model, line_length = 80L)
        if (is.character(model$loss))
          model_info$loss_function <- model$loss
        else if (inherits(model$loss, "python.builtin.function"))
          model_info$loss_function <- model$loss$`__name__`
        optimizer <- model$optimizer
        if (!is.null(optimizer)) {
          model_info$optimizer <- py_str(optimizer)
          model_info$learning_rate <- as.double(optimizer$lr)
        }
        tfruns::write_run_metadata("properties", model_info)
      }, error = function(e) {
        warning("Unable to log model info: ", e$message, call. = FALSE)
      })

    }
  )
)
