#!/usr/bin/env Rscript

envir::attach_source("tools/utils.R")

# options(error = function(e) print(rlang::trace_back()))

resolve_roxy_tether <- function(endpoint) {
  # message("parsing @tether ", endpoint)
  export <- mk_export(endpoint)
  roxy <- export$roxygen |> str_split_lines()
  fn <- as.function(c(formals(export$r_fn), quote({})))

  as_glue(str_flatten_lines(
    str_c("#' ", roxy),
    str_c(export$r_name, " <- "),
    "__signature__",
    format_py_signature(export$py_obj, name = endpoint)
    # deparse(fn)
  ))
}
# resolve_roxy_tether("keras.layers.Dense") |> cat()

# TODO: write out the raw docstring / sig and tutobook to .tether/raw/*, as a safeguard for bugs in the tether resolve functions

# url <- "https://raw.githubusercontent.com/keras-team/keras/master/guides/writing_your_own_callbacks.py"
resolve_rmd_tether <- function(url) {
  path <- url
  path <- sub("https://raw.githubusercontent.com/keras-team/keras/master/",
              "~/github/keras-team/keras/", path, fixed = TRUE)
  path <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              "~/github/keras-team/keras-io/", path, fixed = TRUE)
  tutobook_to_rmd(path, outfile = FALSE)
}
# options(error = browser)

doctether::retether(
  # roxy_tag_eval = NULL,
  roxy_tag_eval = resolve_roxy_tether,
  rmd_field_eval = resolve_rmd_tether
)


message("DONE!")
stop("DONE")


py_run_string("import keras")

parse_tether_tag <- function(endpoint) {
  py_obj <- py_eval(endpoint)
  roxy <- py_obj$`__doc__` |> glue::trim()
  fn <- function() {}
  formals(fn) <- formals(py_obj)  # just the signature

  paste0(collapse = "\n",
    paste0("#' ", roxy),
    deparse(fn)
  )
}

# get_tether("keras.layers.Dense") |> cat()

doctether::retether(tag_parser = parse_tether_tag)


if(FALSE) {


  system2("code", c(
    "-n",
    fs::path(
      dirname(keras$`__path__`),
      keras$layers$Dense$`__module__`  |>
        str_replace_all(fixed('.'), '/'),
      ext = "py"
    )

  ))
  # file.edit('/Users/tomasz/.virtualenvs/r-keras/lib/python3.10/site-packages/keras/src/layers/core/dense.py')
  # file.edit(paste0(keras$layers$Dense$`__module__`, ".py"))

  edit(keras$layers$Dense$`__doc__`)



  if(!"source:tools/utils.R" %in% search())
    envir::attach_source("tools/utils.R")

  if(FALSE) {

    keras$utils # force keras load
    local({
      `__main__` <- reticulate::import_main()
      `__main__`$keras <- keras
      `__main__`$keras_core <- keras
      `__main__`$tf <- tf
      `__main__`$tensorflow <- tf
    })



    get_endpoint_tether_doc <- function(endpoint) {
      as_glue(str_flatten_lines(
        endpoint,
        "__doc__",
        get_docstring(endpoint),
        "__signature__",
        format_py_signature(endpoint)
      ))
    }


    str_flatten_lines <- function(..., na.rm = FALSE) {
      stringr::str_flatten(unlist(c(...)), na.rm = na.rm, collapse = "\n")
    }

  }

}

  # parse_env_setup = function(e) { }
