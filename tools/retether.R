#!/usr/bin/env Rscript

envir::attach_source("tools/utils.R")
library(doctether)

resolve_roxy_tether <- function(endpoint) {
  tryCatch({


  tether <- as_tether(py_obj <- py_eval(endpoint),
                      name = endpoint,
                      roxify = FALSE)
  if(inherits(py_obj, "python.builtin.module")) {
    attr(tether, "roxified") <- NA
    return(tether)
  }
  export <- mk_export(endpoint)
  attr(tether, "roxified") <- str_flatten_lines(
    str_c("#' ", str_split_lines(export$roxygen)),
    deparse(export$r_fn)
  )
  tether
  }, error = function(e) {

  message("endpoint <- ", glue::double_quote(endpoint))
  })
}

resolve_roxy_tether("keras.ops")

# endpoint <- 'keras.layers.BatchNormalization'
# resolve_roxy_tether(endpoint)
#
# stop()
# x <- resolve_roxy_tether('keras.layers.Conv1D')
# cat(x)
# cat(attr(x, "roxified"))


# url <- "https://raw.githubusercontent.com/keras-team/keras/master/guides/writing_your_own_callbacks.py"
resolve_rmd_tether <- function(url) {
  path <- url
  path <- sub("https://raw.githubusercontent.com/keras-team/keras/master/",
              "~/github/keras-team/keras/", path, fixed = TRUE)
  path <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              "~/github/keras-team/keras-io/", path, fixed = TRUE)
  tutobook_to_rmd(path, outfile = FALSE)
}

resolve_rmd_tether <- NULL

# debug(Filter)
# debugonce(doctether:::get_block_name)
# debugonce(doctether::retether)
options(warn = 2)
# options(error = browser)
# debug(roxygen2:::warn_roxy_tag)
# unlink(".tether", recursive = TRUE)
doctether::retether(
  "keras.ops",
  unsafe = TRUE,
  roxy_tag_eval = resolve_roxy_tether,
  # roxy_tag_eval = NULL,
  rmd_field_eval = resolve_rmd_tether
  # rmd_field_eval = NULL
)

# to retether just one vignette:
# doctether:::retether_rmd("~/github/rstudio/keras/vignettes-src/serialization_and_saving.Rmd",
#               eval_tether_field = resolve_rmd_tether)


message("DONE!")
