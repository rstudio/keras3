


callback_history_viewer <- function() {
  
  R6::R6Class("KerasHistoryViewer",
              
    inherit = KerasCallback,
    
    public = list(
      
      metrics = list(),
      
      history_viewer = NULL,
      
      on_epoch_end = function(epoch, logs = NULL) {
      
        # record metrics
        for (metric in names(logs))
          self$metrics[[metric]] <- c(self$metrics[[metric]], logs[[metric]])
        
        # create history object
        history <- keras_training_history(self$params, self$metrics)
        
        # create the history_viewer or update if we already have one
        if (epoch > 0) {
          if (is.null(self$history_viewer)) {
            self$history_viewer <- view_history(history)
          }
          else {
            update_history(self$history_viewer, history)
          }
          
          # pump events
          Sys.sleep(0.2)
        }
      },
      
      on_train_end = function(logs = NULL) {
        
        # write a static version of the history at the end of training
        # (enables saving & publishing of the history)
        if (!is.null(self$history_viewer)) {
          write_static_history(self$history_viewer, 
                               keras_training_history(self$params, self$metrics))
        }
      }
    )
  )$new()
}





view_history <- function(history) {
  
  # create a new temp directory for the viewer's UI/data
  viewer_dir <- tempfile("keras-metrics")
  dir.create(viewer_dir, recursive = TRUE)
  
  # create the history_viewer instance
  history_viewer <- structure(class = "keras_history_viewer", list(
    viewer_dir = viewer_dir
  ))
  
  # copy dependencies to the viewer dir
  history_viewer_html <- system.file("history_viewer", package = "keras")
  file.copy(from = list.files(history_viewer_html, full.names = TRUE),
            to = viewer_dir)
  
  # write the history
  update_history(history_viewer, history)
  
  # view it
  viewer <- getOption("viewer")
  metrics <- Filter(function(name) !grepl("^val_", name), history$params$metrics)
  height <- length(metrics) * 250
  viewer(file.path(viewer_dir, "index.html"), height = height)
  
  # return history_viewer instance (invisibly) for subsequent
  # calls to update_run_history
  invisible(history_viewer)
}


# update the history
update_history <- function(history_viewer, history) {
  history_json <- file.path(history_viewer$viewer_dir, "history.json")
  jsonlite::write_json(unclass(history), history_json)
}

write_static_history <- function(history_viewer, history) {
  
  # create json version of history
  history_json <- jsonlite::toJSON(unclass(history))
  
  # substitute static json into template
  history_html <- file.path(history_viewer$viewer_dir, "index.html")
  history_html_lines <- readLines(history_html, encoding = "UTF-8")
  history_html_lines <- sprintf(history_html_lines, history_json)
  
  # re-write with static data
  writeLines(history_html_lines, history_html)
  
}


# check if the current environment can view training history
# (requires the RStudio Viewer for realtime updates)
can_view_history <- function() {
  !is.null(getOption("viewer")) && nzchar(Sys.getenv("RSTUDIO"))
}


