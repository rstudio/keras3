#!/usr/bin/env Rscript

envir::attach_source("tools/utils.R")
library(doctether)

resolve_roxy_tether <- function(endpoint) {
  tryCatch({
    tether <- as_tether(py_obj <- py_eval(endpoint),
                        name = endpoint,
                        roxify = FALSE)
    if (inherits(py_obj, "python.builtin.module")) {
      attr(tether, "roxified") <- NA
      return(tether)
    }
    export <- mk_export(endpoint)
    attr(tether, "roxified") <- str_flatten_lines(str_c("#' ", str_split_lines(export$roxygen)),
                                                  deparse(export$r_fn))
    tether
  }, error = function(e) {
    message("endpoint <- ", glue::double_quote(endpoint))
  })
}


# resolve_roxy_tether("keras.ops")
#
# endpoint <- 'keras.layers.BatchNormalization'
# x <- resolve_roxy_tether(endpoint)
# x
# cat(x)
# cat(attr(x, "roxified"))
# stop()


# url <- "https://raw.githubusercontent.com/keras-team/keras/master/guides/writing_your_own_callbacks.py"
# withr::with_dir("~/github/keras-team/keras-io/", system("git pull"))
# withr::with_dir("~/github/keras-team/keras/", system("git pull"))
resolve_rmd_tether <- function(url) {
  path <- url
  path <- sub("https://raw.githubusercontent.com/keras-team/keras/master/",
              "~/github/keras-team/keras/", path, fixed = TRUE)
  path <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              "~/github/keras-team/keras-io/", path, fixed = TRUE)

  # path <- sub("~/github/keras-team/keras/",
  #             "~/github/keras-team/keras-io/", path, fixed = TRUE)
  tutobook_to_rmd(path, outfile = FALSE)
}

# resolve_rmd_tether <- NULL
# resolve_roxy_tether <- NULL


# options(warn = 2)
doctether::retether(
  # "keras.ops",
  # unsafe = TRUE,
  roxy_tag_eval = resolve_roxy_tether,
  rmd_field_eval = resolve_rmd_tether
)

## To retether just one vignette:
# doctether:::retether_rmd("~/github/rstudio/keras/vignettes-src/serialization_and_saving.Rmd",
#               eval_tether_field = resolve_rmd_tether)


message("DONE!")

if(FALSE) {
  mk_export("keras.optimizers.Lamb")$dump |> cat_cb()
}


view_vignette_adaptation_diff <- function(rmd_file) {
  path <- tether_field <- rmarkdown::yaml_front_matter(rmd_file)$tether
  path <- sub("https://raw.githubusercontent.com/keras-team/keras/master/",
              "~/github/keras-team/keras/", path, fixed = TRUE)
  path <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              "~/github/keras-team/keras-io/", path, fixed = TRUE)

  # tether_file <- doctether:::get_tether_file(rmd_file)
  tether_file <- path
  system2("code", c("--diff", tether_file, rmd_file))
}

# view_vignette_adaptation_diff("vignettes-src/writing_a_training_loop_from_scratch.Rmd")
