# defered TODOs:
# TODO: consider using quilt or stgit instead of the manual git calls
# https://stacked-git.github.io   https://savannah.nongnu.org/projects/quilt/ https://blog.tfnico.com/2020/07/git-tools-for-keeping-patches-on-top-of.html




# filter out function handles that also have class handles
endpoints <- endpoints %>%
  lapply(\(endpoint) {
    if(!any(startsWith(endpoint, c("keras.losses.", "keras.metrics."))))
      return(endpoint)

    py_obj <- py_eval(endpoint)
    if (!inherits(py_obj, "python.builtin.function"))
      return(endpoint)


    class_endpoint <- switch(endpoint,
                             "keras.losses.kl_divergence" = "keras.losses.KLDivergence",
                             str_replace(endpoint, py_obj$`__name__`,
                                         snakecase::to_upper_camel_case(py_obj$`__name__`)))
    tryCatch({
      py_eval(class_endpoint)
      return(NULL)
    },
    python.builtin.AttributeError = function(e) {
      # don't emit warning about known function handles without class handle
      # counterparts
      if (!endpoint %in% sprintf(
        "keras.metrics.%s", c(
          "binary_focal_crossentropy",
          'categorical_focal_crossentropy',
          'huber',
          'kl_divergence',
          'log_cosh')))
        # browser()
        print(e)
      endpoint
    })
  }) %>%
  unlist() %>%
  invisible()

get_metric_counterpart_endpoint <- function(..., fn, type) {
  if(missing(fn)) {
    endpoint <- type
    py_obj <- py_eval(endpoint)
    name <- py_obj$`__name__`
    endpoint_fn <- str_replace(endpoint, name, snakecase::to_snake_case(name))
    py_obj_fn <- py_eval(endpoint_fn) %error% browser()
  }
}
