
# this file is evaluated in baseenv()
Sys.setenv(CUDA_VISIBLE_DEVICES = "")

# register fake @tether tag parser for roxygen2
local({
  register_tether_tag_parser <- function(...) {
    # message("Registering @tether tag parser")
    registerS3method(genname = "roxy_tag_parse",
                     class =   "roxy_tag_tether",
                     method = identity, #as.function(alist(x = , x), envir = baseenv()),
                     envir = asNamespace("roxygen2"))
  }
  if (isNamespaceLoaded('roxygen2')) register_tether_tag_parser()
  else setHook(packageEvent("roxygen2", "onLoad"), register_tether_tag_parser)
})

local({

  on_py_init <- function() {

    # silence useless warnings from tensorflow
    reticulate:::py_register_load_hook("tensorflow", function() {

      if(keras3:::is_mac_arm64())
      reticulate::py_run_string(local = TRUE, glue::trim(r"---(
        import tensorflow as tf
        tf.config.set_visible_devices(tf.config.get_visible_devices("CPU"))
        )---"
      ))
        # tf$config$get_visible_devices("CPU") |>
        # tf$config$set_visible_devices()


      reticulate::py_run_string(local = TRUE, glue::trim(r"---(
        from importlib import import_module
        import tensorflow as tf

        m = import_module(tf.function.__module__)
        m.FREQUENT_TRACING_WARNING_THRESHOLD = float("inf")
        )---"
      ))
    })

    # R CMD check: no unicode in .Rd files for PDF manuals
    reticulate:::py_register_load_hook("keras", function() {
      reticulate::py_run_string(local = TRUE, glue::trim(r"---(
        def patch_rich_Table_ascii_only():
            from functools import wraps

            from rich.table import Table
            from rich.box import ASCII_DOUBLE_HEAD

            og_Table__init__ = Table.__init__

            @wraps(Table.__init__)
            def __init__(self, *args, **kwargs):
                kwargs["safe_box"] = True
                kwargs["box"] = ASCII_DOUBLE_HEAD
                return og_Table__init__(self, *args, **kwargs)

            Table.__init__ = __init__

        patch_rich_Table_ascii_only()
        )---"
      ))
    })

  }

  on_reticulate_load <- function(...) {
    if (reticulate::py_available()) on_py_init()
    else setHook("reticulate.onPyInit", on_py_init)
  }
  if (isNamespaceLoaded('reticulate')) on_reticulate_load()
  else setHook(packageEvent("reticulate", "onLoad"), on_reticulate_load)

  on_keras3_load <- function(...) {
    keras3::use_backend("tensorflow", gpu = FALSE)
  }
  if (isNamespaceLoaded('keras3')) on_keras3_load()
  else setHook(packageEvent("keras3", "onLoad"), on_keras3_load)

})

# setup knitr hooks for roxygen rendering block example chunks
local({

  # roxygen2 creates one evalenv per block, then calls knit() once per chunk
  process_chunk_output <- function(x, options) {
    # this hook get called with each chunk output.
    # x is a single string of collapsed lines, terminated with a final \n
    final_new_line <- endsWith(x[length(x)], "\n")
    x <- x |> strsplit("\n") |> unlist() |> trimws("right")

    # strip object addresses; no noisy diff
    x <- sub(" at 0[xX][0-9A-Fa-f]{9,16}>", " at 0x0>", x, perl = TRUE)

    # remove reticulate hint from exceptions
    x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
    x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]

    # need ascii only
    # this is used in model summary to indicate layers in a nested model
    x <- gsub("\u2514", ">", x, fixed = TRUE)

    x <- paste0(x, collapse = "\n")
    if (final_new_line && !endsWith(x, "\n"))
      x <- paste0(x, "\n\n")
    x
  }

  # we delay setting the output hook `knit_hooks$set(output = )` because
  # if we set it too early, knitr doesn't set `render_markdown()` hooks.
  # so we set a chunk option, which triggers setting the output hook
  # after knitr is already setup and knitting.
  knitr_on_load <- function() {

    knitr::opts_hooks$set(

      keras.roxy.post_process_output = function(options) {
        # this is a self destructing option, run once before the first
        # chunk in a roxy block is evaluated. Though, with the way roxygen2
        # evaluates blocks currently, this serves no real purpose,
        # since each chunk is an independent knit() call with opts_chunk reset.
        options$keras.roxy.post_process <- NULL
        knitr::opts_chunk$set(keras.roxy.post_process = NULL)

        # make output reproducible
        # `evalenv` is created once per block, but knit() is called once per chunk
        # so we use this to detect if we're in the first chunk of a block and run setup
        if (is.null(roxygen2::roxy_meta_get("evalenv")$.__ran_keras_block_init__)) {
          options(width = 76)
          keras3::clear_session()
          keras3::set_random_seed(1L)
          assign(x = ".__ran_keras_block_init__",
                 envir = roxygen2::roxy_meta_get("evalenv"),
                 value = TRUE)
        }

        local({
          # set the output hook
          og_output_hook <- knitr::knit_hooks$get("output")
          if (identical(attr(og_output_hook, "name", TRUE),
                        "keras.roxy.post_process_output")) {
            # the hook is already set (should never happen,
            # since roxygen calls knit() once per chunk)
            return()
          }

          knitr::knit_hooks$set(output = structure(
            name = "keras.roxy.post_process_output",
            function(x, options) {
              x <- process_chunk_output(x, options)
              og_output_hook(x, options)
            }))
        })

        options
      }
    )
  }

  if (isNamespaceLoaded('knitr')) knitr_on_load()
  else setHook(packageEvent("knitr", "onLoad"), knitr_on_load)

})

list(
  markdown = TRUE,
  r6 = FALSE,
  # packages = "doctether",
  # if we had this, then wouldn't need to register
  # the fake tether tag parser.
  roclets = c("namespace", "rd"),
  knitr_chunk_options = list(
    comment = "##",
    collapse = FALSE,
    eval = !interactive(), # for faster interactive workflow
    keras.roxy.post_process_output = TRUE
  )
)
