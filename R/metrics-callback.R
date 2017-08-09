

KerasMetricsCallback <- R6::R6Class("KerasMetricsCallback",
                                  
  inherit = KerasCallback,
  
  public = list(
    
    metrics = list(),
    
    metrics_viewer = NULL,
    
    on_train_begin = function(logs = NULL) {
      
      # initialize metrics
      for (metric in self$params$metrics)
        self$metrics[[metric]] <- numeric()
      
      # update view
      self$update_view(logs)
      
      # pump events
      Sys.sleep(0.5)
    },
    
    on_epoch_end = function(epoch, logs = NULL) {
      
      # update view
      self$update_view(logs)
      
      # pump events
      Sys.sleep(0.1)
    },
    
    update_view = function(logs = NULL) {
      
      # record metrics
      for (metric in names(logs))
        self$metrics[[metric]] <- c(self$metrics[[metric]], logs[[metric]])
      
      # create history object and convert to metrics data frame
      history <- keras_training_history(self$params, self$metrics)
      metrics <- self$as_metrics_df(history)
      
      # create the metrics_viewer or update if we already have one
      if (is.null(self$metrics_viewer)) {
        self$metrics_viewer <- tfruns::view_run_metrics(metrics)
      }
      else {
        tfruns::update_run_metrics(self$metrics_viewer, metrics)
      }
    },
    
    # convert keras history to metrics data frame suitable for plotting
    as_metrics_df = function(history) {
      
      # create metrics data frame
      df <- as.data.frame(history$metrics)
      
      # pad if necessary
      pad <- history$params$epochs - nrow(df)
      pad_data <- list()
      for (metric in history$params$metrics)
        pad_data[[metric]] <- rep_len(NA, pad)
      df <- rbind(df, pad_data)
      
      # return df
      df
    }
  )
)

