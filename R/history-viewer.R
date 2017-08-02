



callback_history_viewer <- function() {
  
  R6::R6Class("KerasHistoryViewer",
              
    inherit = KerasCallback,
    
    public = list(
      
      metrics = list(),
      
      history_viewer = NULL,
      
      on_epoch_end = function(epoch, logs = NULL) {
        
        # get params (tweak epochs based on current epoch)
        params <- self$params
       
        # record metrics
        for (metric in names(logs))
          self$metrics[[metric]] <- c(self$metrics[[metric]], logs[[metric]])
        
        # create history object
        history <- keras_training_history(params, self$metrics)
        
        # create the history_viewer or update if we already have one
        if (is.null(self$history_viewer))
          self$history_viewer <- view_history(history)
        else
          update_history(self$history_viewer, history)
      }
    )
  )$new()
}


view_history <- function(history) {
  
  # create a new temp directory for the viewer's UI/data
  viewer_dir <- tempfile("viewhtml")
  dir.create(viewer_dir)
  
  # create the history_viewer instance
  history_viewer <- structure(class = "keras_history_viewer",
    viewer_dir = viewer_dir
  )
  
  # copy dependencies to the viewer dir
  history_viewer_html <- system.file("history_viewer", package = "keras")
  file.copy(from = list.files(history_viewer_html),
            to = viewer_dir)
  
  # write the history
  update_history(history_viewer, history)
  
  # view it
  getOption("viewer")(file.path(viewer_dir, "index.html"))
  
  # return history_viewer instance (invisibly) for subsequent
  # calls to update_run_history
  invisible(history_viewer)
}


# update the history
update_history <- function(history_viewer, history) {
  history_json <- file.path(history_viewer$viewer_dir, "history.json")
  jsonlite::write_json(unclass(history), history_json)
}


# check if the current environment can view training history
# (requires the RStudio Viewer for realtime updates)
can_view_history <- function() {
  !is.null(getOption("viewer")) && nzchar(Sys.getenv("RSTUDIO"))
}


