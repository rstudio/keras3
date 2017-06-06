

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


