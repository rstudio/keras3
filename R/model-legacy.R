fit_generator_legacy <- function(object, generator, steps_per_epoch, epochs = 1,
                          verbose=getOption("keras.fit_verbose", default = 1), callbacks = NULL,
                          view_metrics = getOption("keras.view_metrics", default = "auto"),
                          validation_data = NULL, validation_steps = NULL,
                          class_weight = NULL, max_queue_size = 10, workers = 1, initial_epoch = 0) {

  # resolve view_metrics
  if (identical(view_metrics, "auto"))
    view_metrics <- resolve_view_metrics(verbose, epochs, object$metrics)

  if (is.list(validation_data))
    validation_data <- do.call(reticulate::tuple, keras_array(validation_data))

  history <- call_generator_function(object$fit_generator, list(
    generator = generator,
    steps_per_epoch = as.integer(steps_per_epoch),
    epochs = as.integer(epochs),
    verbose = as.integer(verbose),
    callbacks = normalize_callbacks_with_metrics(view_metrics, initial_epoch, callbacks),
    validation_data = validation_data,
    validation_steps = as_nullable_integer(validation_steps),
    class_weight = as_class_weight(class_weight),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers),
    initial_epoch = as.integer(initial_epoch)
  ))

  # convert to a keras_training history object
  history <- to_keras_training_history(history)

  # write metadata from history
  write_history_metadata(history)

  # return the history invisibly
  invisible(history)
}

evaluate_generator_legacy <- function(object, generator, steps, max_queue_size = 10, workers = 1,
                               callbacks = NULL) {

  args <- list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers)
  )

  args <- resolve_callbacks(args, callbacks)

  # perform evaluation
  result <- call_generator_function(object$evaluate_generator, args)

  # apply names
  names(result) <- object$metrics_names

  # write run data
  tfruns::write_run_metadata("evaluation", result)

  # return result
  result
}

predict_generator_legacy <- function(object, generator, steps, max_queue_size = 10, workers = 1, verbose = 0,
                              callbacks = NULL) {

  args <- list(
    generator = generator,
    steps = as.integer(steps),
    max_queue_size = as.integer(max_queue_size),
    workers = as.integer(workers)
  )

  if (keras_version() >= "2.0.1")
    args$verbose <- as.integer(verbose)

  args <- resolve_callbacks(args, callbacks)

  call_generator_function(object$predict_generator, args)
}

call_generator_function <- function(func, args) {

  # check if any generators should run on the main thread
  use_main_thread_generator <-
    is_main_thread_generator(args$generator) ||
    is_main_thread_generator(args$validation_data)

  # handle generators
  args$generator <- as_generator(args$generator)
  if (!is.null(args$validation_data))
    args$validation_data <- as_generator(args$validation_data)

  # force use of thread based concurrency
  if (keras_version() >= "2.0.6") {
    args$use_multiprocessing <- FALSE
  } else {
    args$max_q_size <- args$max_queue_size
    args$max_queue_size <- NULL
    args$pickle_safe <- FALSE
  }

  # if it's a main thread generator then force workers to correct value
  if (use_main_thread_generator) {

    # error to use workers > 1 for main thread generator
    if (args$workers > 1) {
      stop('You may not specify workers > 1 for R based generator functions (R ',
           'generators must run on the main thread)', call. = FALSE)
    }

    # set workers to 0 for versions of keras that support this
    if (keras_version() >= "2.1.2")
      args$workers = 0L
    else
      args$workers = 1L
  }

  # call the generator
  do.call(func, args)
}
