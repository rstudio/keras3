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

resolve_rmd_tether <- NULL
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
  # mk_export("keras.optimizers.Lamb")$dump |> cat_cb()

  mk_export("keras.layers.MaxNumBoundingBoxes")$dump |> cat_cb()
  mk_export("keras.layers.STFTSpectrogram")$dump |> cat_cb()

  mk_export("keras.activations.sparse_plus")$dump |> cat_cb()
  mk_export("keras.activations.sparsemax")$dump |> cat_cb()
  mk_export("keras.activations.threshold")$dump |> cat_cb()

  mk_export("keras.ops.sparse_plus")$dump |> cat_cb()
  mk_export("keras.ops.sparsemax")$dump |> cat_cb()
  mk_export("keras.ops.threshold")$dump |> cat_cb()

  mk_export("keras.ops.diagflat")$dump |> cat_cb()
  mk_export("keras.ops.unravel_index")$dump |> cat_cb()

  mk_export("keras.layers.Equalization")$dump |> cat_cb()
  mk_export("keras.layers.MixUp")$dump |> cat_cb()

  mk_export("keras.layers.RandAugment")$dump |> cat_cb()
  mk_export("keras.layers.RandomColorDegeneration")$dump |> cat_cb()
  mk_export("keras.layers.RandomColorJitter")$dump |> cat_cb()
  mk_export("keras.layers.RandomGrayscale")$dump |> cat_cb()
  mk_export("keras.layers.RandomHue")$dump |> cat_cb()
  mk_export("keras.layers.RandomPosterization")$dump |> cat_cb()



  mk_export("keras.activations.glu")$dump |> cat_cb()
  mk_export("keras.activations.hard_shrink")$dump |> cat_cb()
  mk_export("keras.activations.celu")$dump |> cat_cb()
  mk_export("keras.activations.hard_tanh")$dump |> cat_cb()
  mk_export("keras.activations.log_sigmoid")$dump |> cat_cb()
  mk_export("keras.activations.soft_shrink")$dump |> cat_cb()
  mk_export("keras.activations.squareplus")$dump |> cat_cb()
  mk_export("keras.activations.tanh_shrink")$dump |> cat_cb()
  mk_export("keras.config.is_flash_attention_enabled")$dump |> cat_cb()
  mk_export("keras.losses.circle")$dump |> cat_cb()
  mk_export("keras.initializers.STFT")$dump |> cat_cb()
  mk_export("keras.metrics.ConcordanceCorrelation")$dump |> cat_cb()
  mk_export("keras.metrics.PearsonCorrelation")$dump |> cat_cb()


  mk_export("keras.Model.set_state_tree")$dump |> cat_cb()
  mk_export("keras.layers.Solarization")$dump |> cat_cb()
  mk_export("keras.ops.ifft2")$dump |> cat_cb()

  catched <- character()
  catch <- function(...) catched <<- c(catched, "\n\n", ...)
  mk_export("keras.ops.exp2")$dump               |> catch()
  mk_export("keras.ops.inner")$dump               |> catch()

  # mk_export("keras.ops.glu")$dump               |> catch()
  # mk_export("keras.ops.hard_shrink")$dump            |> catch()
  # mk_export("keras.ops.hard_tanh")$dump        |> catch()
  # mk_export("keras.ops.soft_shrink")$dump               |> catch()
  # mk_export("keras.ops.squareplus")$dump                |> catch()
  # mk_export("keras.ops.tanh_shrink")$dump       |> catch()
  # mk_export("keras.ops.celu")$dump               |> catch()

  #
  # mk_export("keras.ops.dot_product_attention")$dump     |> catch()
  # mk_export("keras.ops.histogram")$dump                 |> catch()
  # mk_export("keras.ops.left_shift")$dump                |> catch()
  # mk_export("keras.ops.right_shift")$dump               |> catch()
  # mk_export("keras.ops.logdet")$dump                    |> catch()
  # mk_export("keras.ops.saturate_cast")$dump             |> catch()
  # mk_export("keras.ops.trunc")$dump                     |> catch()


  catched |>  str_flatten_and_compact_lines(roxygen = TRUE) |> cat_cb()
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
