

envir::attach_source("tools/utils.R")

options(error = function(e) print(rlang::trace_back()))


get_tether <- function(endpoint) {
  message("parsing @tether ", endpoint)
  export <- mk_export(endpoint)
  roxy <- export$roxygen |> str_split_lines()
  fn <- as.function(c(formals(export$r_fn), quote({})))

  as_glue(str_flatten_lines(
    str_c("#' ", roxy),
    str_c(export$r_name, " <- "),
    deparse(fn)
  ))
}

# get_tether("keras.layers.Dense") |> cat()

doctether::update_tethers(
  tag_parser = get_tether,
  resolve_tether_file = function(name) {
    fs::path(glue("man-src/tether/{name}.R"))
    # fs::path(glue("man-src/tether/{name}/1-formatted.md"))
  })

message("DONE!")


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
