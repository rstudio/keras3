

#' @export
print.keras_training_history <- function(x, ...) {
  
  # training params
  params <- x$params
  params <- list(samples = params$samples, 
                 validation_samples = params$validation_samples,
                 batch_size = params$batch_size, 
                 epochs = params$epochs)
  params <-  prettyNum(params, big.mark = ",")
  if (!is.null(params[["validation_samples"]]))
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

#' @export
plot.keras_training_history <- function(x, y, metrics = NULL, ...) {
  
  # if metrics is null we plot all of the metrics
  if (is.null(metrics))
    metrics <- Filter(function(name) !grepl("^val_", name), names(x$metrics))
  
  # par
  op <- par(mfrow = c(length(metrics),1),
            mar=c(3,3,2,2)) # (bottom, left, top, right)
  on.exit(par(op), add = TRUE)
  
  for (i in 1:length(metrics)) {
    
    # get metric
    metric <- metrics[[i]]
    
    # adjust margins
    top_plot <- i == 1
    bottom_plot <- i == length(metrics)
    if (top_plot)
      par(mar = c(1.5,3,1.5,1.5)) 
    else if (bottom_plot)
      par(mar = c(2.5,3,.5,1.5))
    else
      par(mar = c(1.5,3,.5,1.5))
    
    # plot values
    epochs <- 1:x$params$epochs
    legend <- c(metric)
    pch <- c(1)
    values <- x$metrics[[metric]]
    plot(epochs, values, xaxt = ifelse(bottom_plot, 's', 'n'),
         xlab = "epoch", ylab = metric, pch = pch[[1]], ...)
    
    # plot validation values if we have them
    val_metric <- paste0("val_", metric)
    val_values <-x$metrics[[val_metric]] 
    if (!is.null(val_values)) {
      pch <- c(pch, 4)
      legend <- c(legend, val_metric)
      points(epochs, val_values, pch = pch[[2]])
    }
   
    # add legend
    legend_location <- ifelse(values[[1]] > values[[x$params$epochs]],
                              "topright", "bottomright")
    legend(legend_location, legend = legend, pch = pch)
  }
}


