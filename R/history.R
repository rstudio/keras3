

#' @export
print.keras_training_history <- function(x, ...) {

  # compute epochs actuually trained for
  epochs <- min(x$params$epochs, length(x$metrics[[1]]))

  # training params
  params <- x$params
  params <- list(samples = params$samples,
                 validation_samples = params$validation_samples,
                 batch_size = params$batch_size,
                 epochs = epochs)
  params <-  prettyNum(params, big.mark = ",")
  if (!identical(params[["validation_samples"]], "NULL"))
    validate <- paste0(", validated on ", params[["validation_samples"]], " samples")
  else
    validate <- ""

  str <- ""
  if (!params[["samples"]] == "NULL") {
    str <- paste0(str, "Trained on ", params[["samples"]]," samples", validate, " (batch_size=",
                  params[["batch_size"]], ", epochs=", params[["epochs"]], ")")
  }

  # last epoch metrics
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
  cat(str, "\n")
}


#' Plot training history
#'
#' Plots metrics recorded during training.
#'
#' @param x Training history object returned from
#'  `fit.keras.engine.training.Model()`.
#' @param y Unused.
#' @param metrics One or more metrics to plot (e.g. `c('loss', 'accuracy')`).
#'   Defaults to plotting all captured metrics.
#' @param method Method to use for plotting. The default "auto" will use
#'   \pkg{ggplot2} if available, and otherwise will use base graphics.
#' @param smooth Whether a loess smooth should be added to the plot, only
#'   available for the `ggplot2` method. If the number of epochs is smaller
#'   than ten, it is forced to false.
#' @param theme_bw Use `ggplot2::theme_bw()` to plot the history in
#'   black and white.
#' @param ... Additional parameters to pass to the [plot()] method.
#'
#' @export
plot.keras_training_history <- function(x, y, metrics = NULL, method = c("auto", "ggplot2", "base"),
                                        smooth = getOption("keras.plot.history.smooth", TRUE),
                                        theme_bw = getOption("keras.plot.history.theme_bw", FALSE),
                                        ...) {
  # check which method we should use
  method <- match.arg(method)
  if (method == "auto") {
    if (requireNamespace("ggplot2", quietly = TRUE))
      method <- "ggplot2"
    else
      method <- "base"
  }

  # convert to data frame
  df <- as.data.frame(x)

  # if metrics is null we plot all of the metrics
  if (is.null(metrics))
    metrics <- Filter(function(name) !grepl("^val_", name), names(x$metrics))

  # select the correct metrics
  df <- df[df$metric %in% metrics, ]

  if (tensorflow::tf_version() < "2.2")
    do_validation <- x$params$do_validation
  else
    do_validation <- any(grepl("^val_", names(x$metrics)))


  if (method == "ggplot2") {
    # helper function for correct breaks (integers only)
    int_breaks <- function(x) pretty(x)[pretty(x) %% 1 == 0]

    if (do_validation) {
      if (theme_bw)
        p <- ggplot2::ggplot(df, ggplot2::aes_(~epoch, ~value, color = ~data, fill = ~data, linetype = ~data, shape = ~data))
      else
        p <- ggplot2::ggplot(df, ggplot2::aes_(~epoch, ~value, color = ~data, fill = ~data))
    } else {
      p <- ggplot2::ggplot(df, ggplot2::aes_(~epoch, ~value))
    }

    smooth_args <- list(se = FALSE, method = 'loess', na.rm = TRUE)

    if (theme_bw) {
      smooth_args$size <- 0.5
      smooth_args$color <- "gray47"
      p <- p +
        ggplot2::theme_bw() +
        ggplot2::geom_point(col = 1, na.rm = TRUE, size = 2) +
        ggplot2::scale_shape(solid = FALSE)
    } else {
      p <- p +
        ggplot2::geom_point(shape = 21, col = 1, na.rm = TRUE)
    }

    if (smooth && x$params$epochs >= 10)
      p <- p + do.call(ggplot2::geom_smooth, smooth_args)

    p <- p +
      ggplot2::facet_grid(metric~., switch = 'y', scales = 'free_y') +
      ggplot2::scale_x_continuous(breaks = int_breaks) +
      ggplot2::theme(axis.title.y = ggplot2::element_blank(), strip.placement = 'outside',
                     strip.text = ggplot2::element_text(colour = 'black', size = 11),
                     strip.background = ggplot2::element_rect(fill = NA, color = NA))

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
      if (do_validation)
        graphics::legend(legend_location, legend = c(metric, paste0("val_", metric)), pch = c(1, 4))
      else
        graphics::legend(legend_location, legend = metric, pch = 1)
    }
  }
}


#' @export
as.data.frame.keras_training_history <- function(x, ...) {

  # filter out metrics that were collected for callbacks (e.g. lr)
  if (tensorflow::tf_version() < "2.2")
    x$metrics <- x$metrics[x$params$metrics]

  if (tensorflow::tf_version() >= "2.1")
    metric_names <- names(x$metrics)
  else
    metric_names <- x$params$metrics

  # pad to epochs if necessary
  values <- x$metrics
  pad <- x$params$epochs - length(values$loss)
  pad_data <- list()
  for (metric in metric_names)
    pad_data[[metric]] <- rep_len(NA, pad)
  values <- rbind(values, pad_data)

  # prepare data to plot as a data.frame
  df <- data.frame(
    epoch = seq_len(x$params$epochs),
    value = unlist(values),
    metric = rep(sub("^val_", "", names(x$metrics)), each = x$params$epochs),
    data = rep(grepl("^val_", names(x$metrics)), each = x$params$epochs)
  )
  rownames(df) <- NULL

  # order factor levels appropriately
  df$data <- factor(df$data, c(FALSE, TRUE), c('training', 'validation'))
  df$metric <- factor(df$metric, unique(sub("^val_", "", names(x$metrics))))

  # return
  df
}

to_keras_training_history <- function(history) {


  # turn history into an R object so it can be persited and
  # and give it a class so we can write print/plot methods
  params <- history$params

  # we only see this info before TF 2.2
  if (tensorflow::tf_version() < "2.2") {
    if (params$do_validation) {
      if (!is.null(params$validation_steps))
        params$validation_samples <- params$validation_steps
      else
        params$validation_samples <- dim(history$validation_data[[1]])[[1]]
    }
  }

  # normalize metrics
  metrics <- history$history
  metrics <- lapply(metrics, function(metric) {
    as.numeric(lapply(metric, mean))
  })

  # create history
  keras_training_history(
    params = params,
    metrics = metrics
  )
}

keras_training_history <- function(params, metrics) {

  # pad missing metrics with NA
  rows <- max(as.integer(lapply(metrics, length)))
  for (metric in names(metrics)) {
    metric_data <- metrics[[metric]]
    pad <- rows - length(metric_data)
    pad_data <- rep_len(NA, pad)
    metric_data <- c(metric_data, pad_data)
    metrics[[metric]] <- metric_data
  }

  # return history
  structure(class = "keras_training_history", list(
    params = params,
    metrics = metrics
  ))
}
