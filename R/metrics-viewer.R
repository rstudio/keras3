

view_run_metrics <- function(metrics) {
  
  # create a new temp directory for the viewer's UI/data
  viewer_dir <- tempfile("keras-metrics")
  dir.create(viewer_dir, recursive = TRUE)
  
  # create the metrics_viewer instance
  metrics_viewer <- structure(class = "tfruns_metrics_viewer", list(
    viewer_dir = viewer_dir
  ))
  
  # copy dependencies to the viewer dir
  metrics_viewer_html <- system.file("metrics_viewer", package = "keras")
  file.copy(file.path(metrics_viewer_html, c(
    "d3.min.js",
    "c3.min.js",
    "c3.min.css",
    "metrics.js",
    "metrics.css")),
    to = viewer_dir)
  
  # write the history
  update_run_metrics(metrics_viewer, metrics)
  
  # view it
  viewer <- getOption("viewer", default = browser_viewer(viewer_dir))
  viewer(file.path(viewer_dir, "index.html"))
  
  # return metrics_viewer instance (invisibly) for subsequent
  # calls to update_run_history
  invisible(metrics_viewer)
}


update_run_metrics <- function(viewer, metrics) {
  
  # re-write index.html with embedded history
  history_json <- jsonlite::toJSON(metrics, dataframe = "columns", na = "null")
  history_html <- system.file("metrics_viewer", "index.html", package = "keras")
  history_html_lines <- readLines(history_html, encoding = "UTF-8")
  history_html_lines <- sprintf(history_html_lines, history_json)
  writeLines(history_html_lines, file.path(viewer$viewer_dir, "index.html"))
  
  # write metrics.json for polling
  history_json <- file.path(viewer$viewer_dir, "metrics.json")
  jsonlite::write_json(metrics, history_json, dataframe = "columns", na = "null")
}

# non-rstudio viewer function
browser_viewer <- function(viewer_dir) {
  function(url) {
    # determine help server port
    port <- tools::startDynamicHelp(NA)
    if (port <= 0)
      port <- tools::startDynamicHelp(TRUE)
    if (port <= 0) {
      warning("Unable to view keras training history ",
              "(couldn't access help server port)",
              call. = FALSE)
      return(invisible(NULL))
    }
    
    # determine path to history html
    path <- paste("/session", basename(viewer_dir), "index.html", sep = "/")
    
    # build URL and browse it
    url <- paste0("http://127.0.0.1:", port, path)
    utils::browseURL(url)
  }
}



KerasMetricsViewer <- R6::R6Class("KerasMetricsViewer",
                                  
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
        self$metrics_viewer <- view_run_metrics(metrics)
      }
      else {
        update_run_metrics(self$metrics_viewer, metrics)
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

