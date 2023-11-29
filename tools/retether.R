#!/usr/bin/env Rscript

envir::attach_source("tools/utils.R")
library(doctether)

resolve_roxy_tether <- function(endpoint) {
  tether <- as_tether(py_eval(endpoint),
                      name = endpoint,
                      include_roxified = FALSE)
  export <- mk_export(endpoint)
  attr(tether, "roxified") <- str_flatten_lines(
    str_c("#' ", export$roxygen),
    deparse(export$r_fn)
  )
  tether
}


# url <- "https://raw.githubusercontent.com/keras-team/keras/master/guides/writing_your_own_callbacks.py"
resolve_rmd_tether <- function(url) {
  path <- url
  path <- sub("https://raw.githubusercontent.com/keras-team/keras/master/",
              "~/github/keras-team/keras/", path, fixed = TRUE)
  path <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              "~/github/keras-team/keras-io/", path, fixed = TRUE)
  tutobook_to_rmd(path, outfile = FALSE)
}

# options(warn = 2, error = browser)
# unlink(".tether", recursive = TRUE)
doctether::retether(
  roxy_tag_eval = resolve_roxy_tether,
  rmd_field_eval = resolve_rmd_tether
)


message("DONE!")
