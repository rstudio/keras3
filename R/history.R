

#' @export
print.keras_training_history <- function(x, ...) {
  
  # training params
  params <- x$params
  params <- list(samples = params$samples, 
                 validation_samples = params$validation_samples,
                 batch_size = params$batch_size, 
                 epochs = params$epochs)
  params <-  prettyNum(params, big.mark = ",")
  if (!identical(params[["validation_samples"]], "NULL"))
    validate <- paste0(", validated on ", params[["validation_samples"]], " samples")
  else 
    validate <- ""
  str <- paste0("Trained on ", params[["samples"]]," samples", validate, " (batch_size=", 
                params[["batch_size"]], ", epochs=", params[["epochs"]], ")")

  # last epoch metrics
  epochs <- x$params$epochs
  metrics <- lapply(x$metrics, function(metric) {
    metric[[epochs]]
  })
  
  labels <- names(metrics)
  max_label_len <- max(nchar(labels))
  labels <- sprintf(paste0("%", max_label_len, "s"), labels) 
  metrics <- prettyNum(metrics, big.mark = ",", digits = 4, scientific=FALSE)
  str <- paste0(str, "\n",
                "Final epoch (plot to see history):\n",
                paste0(labels, ": ", metrics, collapse = "\n"),
                collapse = "\n")
  cat(str)
}


#' Plot training history
#' 
#' Plots metrics recorded during training. 
#' 
#' @param x Training history object returned from `fit()`.
#' @param y Unused.
#' @param metrics One or more metrics to plot (e.g. `c('loss', 'accuracy')`).
#'   Defaults to plotting all captured metrics.
#' @param method Method to use for plotting. The default "auto" will use 
#'   \pkg{ggplot2} if available, and otherwise will use base graphics.
#' @param smooth Whether a loess smooth should be added to the plot, only 
#'   available for the `ggplot2` method. If the number of epochs is smaller
#'   than ten, it is forced to false.
#' @param ... Additional parameters to pass to the [plot()] method.
#'
#' @export
plot.keras_training_history <- function(x, y, metrics = NULL, method = c("auto", "ggplot2", "base"), 
                                        smooth = TRUE, ...) {
  # check which method we should use
  method <- match.arg(method)
  if (method == "auto") {
    if (requireNamespace("ggplot2", quietly = TRUE))
      method <- "ggplot2"
    else
      method <- "base"
  }
  
  # if metrics is null we plot all of the metrics
  if (is.null(metrics))
    metrics <- Filter(function(name) !grepl("^val_", name), names(x$metrics))

  # prepare data to plot as a data.frame
  df <- data.frame(
    epoch = seq_len(x$params$epochs),
    value = unlist(x$metrics),
    metric = rep(sub("^val_", "", names(x$metrics)), each = x$params$epochs),
    data = rep(grepl("^val_", names(x$metrics)), each = x$params$epochs)
  )
  
  # select the correct metrics
  df <- df[df$metric %in% metrics, ]
  
  # order factor levles appropriately
  df$data <- factor(df$data, c(FALSE, TRUE), c('training', 'validation'))
  df$metric <- factor(df$metric, unique(sub("^val_", "", names(x$metrics))))
  
  if (method == "ggplot2") {
    # helper function for correct breaks (integers only)
    int_breaks <- function(x) pretty(x)[pretty(x) %% 1 == 0]
    
    if (x$params$do_validation)
      p <- ggplot2::ggplot(df, ggplot2::aes_(~epoch, ~value, color = ~data, fill = ~data))
    else 
      p <- ggplot2::ggplot(df, ggplot2::aes_(~epoch, ~value))
    
    if (smooth && x$params$epochs >= 10)
      p <- p + ggplot2::geom_smooth(se = FALSE, method = 'loess')
    
    p <- p +
      ggplot2::geom_point(shape = 21, col = 1) +
      ggplot2::facet_grid(metric~., switch = 'y', scales = 'free_y') +
      ggplot2::scale_x_continuous(breaks = int_breaks) +
      ggplot2::theme(axis.title.y = ggplot2::element_blank(), strip.placement = 'outside',
                     strip.text = ggplot2::element_text(colour = 'black', size = 11),
                     strip.background = ggplot2::element_rect(fill = NA))
    return(p)
  }
  
  if (method == 'base') {
    # par
    op <- par(mfrow = c(length(metrics), 1),
              mar = c(3, 3, 2, 2)) # (bottom, left, top, right)
    on.exit(par(op), add = TRUE)
    
    for (i in seq_along(metrics)) {
      
      # get metric
      metric <- metrics[[i]]
      
      # adjust margins
      top_plot <- i == 1
      bottom_plot <- i == length(metrics)
      if (top_plot)
        par(mar = c(1.5, 3, 1.5, 1.5)) 
      else if (bottom_plot)
        par(mar = c(2.5, 3, .5, 1.5))
      else
        par(mar = c(1.5, 3, .5, 1.5))
      
      # select data for current panel
      df2 <- df[df$metric == metric, ]
      
      # plot values
      plot(df2$epoch, df2$value, pch = c(1, 4)[df2$data],
           xaxt = ifelse(bottom_plot, 's', 'n'), xlab = "epoch", ylab = metric, ...)
      
      # add legend
      legend_location <- ifelse(
        df2[df2$data == 'training', 'value'][1] > df2[df2$data == 'training', 'value'][x$params$epochs],
        "topright", "bottomright")
      if (x$params$do_validation)
        graphics::legend(legend_location, legend = c(metric, paste0("val_", metric)), pch = c(1, 4))
      else
        graphics::legend(legend_location, legend = metric, pch = 1)
    }
  }
}

KerasHistoryViewer <- R6::R6Class("KerasHistoryViewer",
                                  
  inherit = KerasCallback,
  
  public = list(
    
    metrics = list(),
    
    history_viewer = NULL,
    
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
      
      # create history object
      history <- keras_training_history(self$params, self$metrics)
      
      # create the history_viewer or update if we already have one
      if (is.null(self$history_viewer)) {
        self$history_viewer <- view_history(history)
      }
      else {
        update_history(self$history_viewer, history)
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
)

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
  viewer(file.path(viewer_dir, "index.html"))
  
  # return history_viewer instance (invisibly) for subsequent
  # calls to update_run_history
  invisible(history_viewer)
}


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


keras_training_history <- function(params, metrics) {
  structure(class = "keras_training_history", list(
    params = params,
    metrics = metrics
  ))
}
