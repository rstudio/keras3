#!/usr/bin/env Rscript

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")
if(!"source:tools/translate-tools.R" %in% search()) envir::attach_source("tools/translate-tools.R")

## Deferred:

# TODO: k_fft() should really accept and return complex tensors too.
# TODO: activation_resolve() or activation_get() as alias of keras.activation.get() ?
# TODO: the 2-translated.Rmd should include a chunk w/ the function def (for easy seeing while editing)
#       with chunk options (include = FALSE)


## Rejected:

# TODO: k_array() should take a 'shape' argument

## Implemented but Questioning:

# TODO: maybe move ops to `op_*` instead of `k_*` ?
if(FALSE) {
  Sys.glob(c("R/autogen*.R", "vignettes-src/*.Rmd")) %>%
    walk(\(f) {
      readLines(f) %>%
        str_replace_all("(?<![a-zA-Z0-9])k_", "op_") %>%
        writeLines(f)
    })
  f <- Sys.glob(".tether/man/k_*")
  f2 <- sub("/k_", "/op_", f, fixed = TRUE)
  fs::file_move(f, f2)
}

## In progress


## Waiting to be processed:


# TODO: remove k_amax() and friends, they're redundant w/ k_max(), which already
#       takes an axis arg. Only there for numpy api compatability, which
#       doesn't matter to us.

# TODO: train_on_batch and related methods should be autogen'd and exported. Or maybe we curate those,
#       and don't export them? (I.e., have the few people that need them access methods via model$train_on_batch())

# TODO: Clean up @family tags (partially autogen, derive from module - then use that to autogenerate _pkgdown.yml)

# TODO:  self$model$stop_training <- TRUE should work. Need to avoid propogating `$<-` past first.

# TODO: new_callback_class() should wrap callables to make epoch/batch n 1 based,
#       make logs persistent (e.g., wrap the user callable w/ `logs$update(<callback_return>`))

# TODO: in reticulate, change subclassed dict autoconversion back to off:
#      so that keras$utils$get_custom_objects()$clear() works.

# TODO: get_custom_objects() needs thinking

# TODO: r_name autogen: move "set" to tail, so have config_floatx(), config_floatx_set()

# TODO: @param ... Passed on to the Python callable - scrub this from formatted.md
if(FALSE)
Sys.glob(c("R/autogen*.R", "vignettes-src/*.Rmd")) %>%
  walk(\(f) {
    readLines(f) %>%
      replace_val("#' Passed on to the Python callable", "#' For forward/backward compatability.") %>%
      writeLines(f)
  })
#' Passed on to the Python callable
#' For forward/backward compatability.


# TODO: revisit history - also mentions in docs (e.g., in callback_model_checkpoint())

# TODO: BackupAndRestore is broken, doesn't respect current epoch. file/fix upstream.
#       ~/github/keras-team/keras/keras/callbacks/backup_and_restore_callback.py

# TODO: swap arg order in k_vectorized_map

# TODO: reticulate, support NO_COLOR (or similar) to disable the link wrapper around `py_last_error()` hint.

# TODO: # fix `fit()` not returning `history` correctly

# TODO: add PR for purrr::rate_throttle("3 per minute")
#
# TODO: global search for "axis" in doc text, update to 1 based where appropriate.
#
# TODO: in reticulate: virtualenv_starter(): check for --enable-shared

# TODO: fix py_func(), for r_to_py.R6ClassGenerator
#   can't use __signature__ anymore in keras_core...

## TODO: "keras.applications.convnext" is a module, filtered out has good stuff

# TODO: initializer families:
# <class 'keras.initializers.constant_initializers.Zeros'>
# <class 'keras.initializers.random_initializers.RandomUniform'>

# TODO: next: losses, metrics, saving, guides/vignettes
#
# TODO: global search replace in man-src/*.Rmd "([^ ])=([^ ])" "\\1 = \\2"

# TODO: bidirectional, time_distributed -- need special caseing
#
# TODO: note in docs for k_logical_and (and friends) that these are dispatched
#       to from & != and so on.
#
# TODO: k_arange: should it default to produce floats?

# TODO: keras.Function ?? keras.Variable ?? keras.name_scope ??
#
# TODO: remove k_random_binomial() ??
#
# TODO: layer_feature_space() needs massaging.
#
# TODO: to_categorical():
#    - handle factor/character https://github.com/rstudio/keras/issues/1055
#    - make it 1 based?
#
# TODO: param descriptions - make it more robust to changes upstream
#     autoinject "see description" without needing it in the yml.
#     yml is only for explicit overrides
#
# TODO: implement and export as_shape(), make k_shape() a little nicer (e.g, an integer w/ NA)
#
# TODO: implement dim() S3 generic.
#
# TODO: remove @import methods ??
#
# TODO: add @import reticulate ??
#
# TODO: remove any tensorflow imports / DESCRIPTION deps
#
# TODO: trimws @returns
#
# TODO: k_istft k_irfft example is wrong, investigate
#
# TODO: rename: k_image_pad_images -> k_image_pad
#
# TODO: shape accessor for x$shape?
#
# TODO: this should work: k_convert_to_tensor(c(1, 3, 2, 0), "int32")
#
# TODO: as_0_based_index() utility: as_integer(x + 1)
#
# TODO: k_array(<r_array>) should default to float32, not float64
#
# TODO: k_array(<r_int>) why int64, and not int32?
#
# TODO: revisit docs for k_scatter_update and k_scatter, remove python sliceisms

# TODO: fix k_vectorized_map() arg rename kludge
#
# TODO: layer_category_encoding()(count_weights) call arg example not working
#
# TODO: revisit k_vectorized_map() man page
# The source of truth for the current translation should be...?
#    - the autogened file R/autogen-*.R, or
#    - man-src/*/2-translated.Rmd
#
# TODO: `axis` arg in merging layers has to wrong transformer, should be `as_axis()`, is `as_integer()`

# TODO: get rid of this in params: @param ... Passed on to the Python callable

# TODO: global doc search for None/True/False/[Tt]uple/[Dd]ict(ionary|\\b)
#
# TODO: layer_feature_space() needs many helpers for float_normalized() and friends
#       output_mode = 'dict' should be 'named list' ?
#
# TODO: feature_space saving errors
#
# TODO: layer_lambda docs: bring back section on serialization and limitations after
#       fixing serialization.
#
# TODO: write a .git_reset() helper that restores everything except tools/*
#
# TODO: a layer_hashed_crossing() instance should have a customized wrapper that
#       splices in the args, so instead of layer(list(a, b)), you can do layer(a, b)
#       also, the example should maybe be nicer, with a named list/dict being passed,
#       instead of a tuple
#
# TODO: this shouldn't error (empty last arg should be no-op):
# if(FALSE) {
# keras$utils$FeatureSpace$cross(
#   feature_names=c("string_values", "int_values"),
#   crossing_dim=32,
#   output_mode="int",
# )
# }
#
# TODO: layer_torch_module_wrapper raises an error - aybe incompatible torch version?a
#
# TODO: config_{enable,disable,is_enabled}_traceback_filtering have identical docstrings,
# should all be the same page.
#
# TODO: @family tags should be manually managed, not dynamically generated.
#       perhaps in a yml file
#
# TODO: refactor so 'endpoint' can be a tuple like "keras.losses.Hinge,keras.losses.hinge"
#       and 0-upstream.md is a concatenation of multiple endpoints, separated
#       by a delimiter so we can do
#       read_file("0-upstream.md") |>
#         strsplit(str_c(strrep("~", 80), "\n")) |>
#         lapply(format_upstream)


memoise::forget(mk_export)

get_translations <- function() {
  dirs <- fs::dir_ls("man-src/", regexp = "\\.Rmd$") |>
    sort() %>%
    # .[grep("^k_", basename(.))] %>%
    # .[grep("^layer_", basename(.))] %>%
    set_names(basename) %>%
    # keep(\(dir) read_file(path(dir, "2-translated.Rmd")) |>
    keep(\(dir) read_file(path(dir)) |>
           str_detect("```python")) |>
    (\(x) { message("remaining: ", length(x)); x})() |>
    # (\(x) {walk(x, message); x})()
    head(10) |>
    purrr::walk(\(dir) {
      og <-  read_file(dir)
      # og <-  read_file(dir/"2-translated.Rmd")
      new <- og %>%
        str_split_lines() %>%
        str_replace_all("```python", "```{r}") %>%
        str_replace_all(fixed("keras.ops."), "k_") %>%
        str_replace_all(fixed("ops."), "k_") %>%
        str_replace_all(fixed("= np.array"), "<- k_array") %>%
        str_replace_all(fixed("np.array"), "k_array") %>%
        str_replace_all(fixed(").reshape("), "|> k_reshape(c(") %>%
        str_replace_all(fixed("None"), "NULL") %>%
        str_replace_all(fixed("k_convert_to_tensor(["), "k_convert_to_tensor(c(") %>%
        str_replace_all(fixed("k_array(["), "k_array(c(") %>%
        str_replace_all(fixed("])"), "))") %>%
        str_replace_all(fixed("np.random.rand"), "random_uniform(c") %>%
        str_replace_all("^([a-z_0-9A-Z]+) =", "\\1 <-") %>%
        str_replace_all("None", "NULL") %>%
        str_replace_all("\\bdict(ionary)?", "named list") %>%
        str_replace_all("\\bDict(ionary)?", "Named list") %>%
        str_replace_all("tuple", "list") %>%
        str_replace_all("Tuple", "List") %>%
        str_replace_all("True", "TRUE") %>%
        str_replace_all("False", "FALSE") %>%
        str_replace_all(fixed("np.random.random(("), "random_uniform(c(") %>%
        # str_replace_all("([0-9])(\\.0?\\b)", "\\1") %>%
        str_replace_all(fixed("list/list"), "list") %>%
        str_replace_all(fixed("list/list", ignore_case = TRUE), "List") %>%
        str_replace_all(fixed("keras.layers."), "layer_") %>%
        str_replace_all(fixed("layers."), "layer_") %>%
        str_flatten_lines() %>%
        identity()
        # str_replace_all(fixed("k_convert_to_tensor(["), "k_array(c(") %>%
      # new |> write_lines(dir/"2-translated.Rmd")
      new |> write_lines(dir)
      file.edit(dir)
      # file.edit(dir/"2-translated.Rmd")
      # stop()
      return()
      withr::local_dir(dir)
      message("Translating: ", basename(dir))
      new <- og |> get_translated_roxygen()
      # message("cost: ")
      new |> write_lines("2-translated.Rmd")
      # write_rds(new, "completion.rds")
    })

  x <- system("git diff --name-only", intern = TRUE) %>%
    grep("2-translated.Rmd$", ., value = TRUE)

  x %>%
    double_quote() %>%
    str_flatten(",\n  ") %>%
    str_c("  ", .) %>%
    c("file.edit(", ., ")") %>%
    str_flatten_lines() %>%
    message()
  # message(sprintf("file.edit(%s)", shQuote(dirs))
  # file.edit({str_flatten(x, ', ')})}")
}

if(FALSE) {

  get_translations()

  list.files("man-src", "2-translated\\.Rmd$", recursive = TRUE,
             full.names = TRUE) %>%
    walk(\(f) {
      x <- read_lines(f)
      while(x[1] == "---") {
        x <- x[-(1:3)]
      }
      x <- str_trim(str_flatten_lines(
        "---",
        'knit: ({source(here::here("tools/knit.R")); knit_man_src})',
        "---",
        x))
      write_lines(x, f)
    })

  withr::with_dir("man-src", {
    list.dirs(full.names = TRUE) %>%
      as_fs_path() %>%
      walk(\(d) {
        if (!file_exists(d / "2-translated.Rmd"))
          return()
        # browser()
        link_create(path(d, "2-translated.Rmd"),
                    path(d, ext = "Rmd"))
      })
  })


}

# start by regenerating patch files


endpoints <- list_endpoints(skip = c(
  # to be processed / done
  "keras.saving",
  "keras.backend",
  "keras.dtensor",
  "keras.mixed_precision",
  "keras.models",
  "keras.export",
  "keras.experimental",
  "keras.applications",
  "keras.legacy",
  "keras.distribution",  # multi-host multi-device training
  "keras.protobuf", # ???

  "keras.datasets",            # datasets unchanged, no need to autogen
  "keras.preprocessing.text",  # deprecated
  "keras.estimator",           # deprecated
  "keras.optimizers.legacy",

  "keras.src"                  # internal
)) %>%
  c(list_endpoints("keras.applications", max_depth = 1)) %>%
  # filter out top level non module symbols for now
  grep("keras.([^.]+)$", ., value = TRUE, invert = TRUE) %>%
  unique()

endpoints %<>% filter_out_endpoint_aliases()

## filter out some endpoints that need special handling
endpoints %<>% setdiff(c %(% {
  "keras.layers.Layer"             # only for subclassing
  "keras.optimizers.Optimizer"     # only for subclassing
  "keras.regularizers.Regularizer" # only for subclassing
  "keras.constraints.Constraint"   # only for subclassing
  "keras.initializers.Initializer" # only for subclassing
  "keras.callbacks.Callback"       # only for subclassing
  "keras.losses.Loss"              # only for subclassing
  "keras.metrics.Metric"           # only for subclassing
  "keras.optimizers.schedules.LearningRateSchedule"  # only for subclassing

  "keras.utils.PyDataset"      # parallel processing in R not possible this way
  "keras.utils.Sequence"       # tfdatasets is ~100x better anyway.

  "keras.utils.plot_model"        # S3 method plot()

  # TODO: revisit custom_object_scope()
  "keras.utils.custom_object_scope"  # need custom work to resolve py_names -
  # manually wrapped to `with_custom_object_scope()`

  "keras.metrics.Accuracy"         # weird,
  # only class handle, no fn handle - weird alias
  # for binary_accuracy, but without any threshold casting.
  # kind of confusing - the keras.metrices.<type>*_accuracy
  # endpoints are much preferable.

  "keras.utils.Progbar"            # needs thinking
  "keras.layers.Wrapper"           # needs thinking
  "keras.layers.InputLayer"        # use Input instead
  "keras.layers.InputSpec"         # ??
  # "keras.layers.Bidirectional"         # ??
  "keras.callbacks.CallbackList"   # just an abstract list
  "keras.callbacks.History"        # always added to fit() by default

  "keras.optimizers.LegacyOptimizerWarning"

  "keras.ops.absolute" # alias dup of abs.
})


exports <- endpoints |>
  purrr::set_names() |>
  lapply(mk_export)


df <- exports |>
  lapply(\(e) {
    unclass(e) |> map_if(\(attr) !is_scalar_atomic(attr), list) |>
      as_tibble_row()
  }) |>
  list_rbind() |>
  select(r_name, endpoint, type, module, everything())

df <- df |>
  mutate(
    man_src_dir = path("man-src", r_name),
    endpoint_sans_name = str_extract(endpoint, "keras\\.(.*)\\.[^.]+$", 1))

if(!all(dir_exists(df$man_src_dir))) {
  df |>
    filter(!dir_exists(man_src_dir)) |>
    rowwise() |>
    mutate(init_man_src_dir = {
      # browser()
      man_src_dir |>
        dir_create() |>
        withr::with_dir({
          write_lines(format_man_src_0(endpoint), "0-upstream.md")
          write_lines(roxygen, "1-formatted.md")
          write_lines(roxygen, "2-translated.Rmd")
          NULL
        })
    })
}

if(FALSE) {

df %>%
  filter(r_name |> startsWith("constraint_")) %>%
  rowwise() %>%
  mutate(dump2 = str_flatten_lines(
    str_c("# ", endpoint),
    str_c("#' ", read_lines(path(man_src_dir, "2-translated.Rmd"))),
    str_c("#' ", glue("@tether {endpoint}")),
    str_c(r_name, " <-"),
    deparse(r_fn)
  )) %>%
  ungroup() %>%
  {
    write_lines(str_flatten(.$dump2, "\n\n\n"),
                'R/autogen2-constraints.R')
  }

}

df <- df |>
  arrange(endpoint_sans_name, module, r_name) |>
  mutate(file = if_else(endpoint_sans_name == "layers",
                        {
                          module |>
                            str_replace("^keras(_core)?\\.(src\\.)?", "") |>
                            str_replace(paste0(endpoint_sans_name, "\\."), "") |>
                            str_replace("^([^.]+).*", paste0(endpoint_sans_name, "-\\1.R"))
                        },
                        str_c(endpoint_sans_name %>% str_replace_all(fixed("."), "-"), ".R"))
  )

unlink(Sys.glob("R/autogen-*.R"))


get_translated_lines <- function(r_name) {
  x <- readLines(fs::path("man-src", r_name, "2-translated.Rmd"))
  if(x[1] == "---" && x[3] == '---')
    x <- x[-(1:3)]
  x <- str_trim(x, "right")
}

stop("Edit files in R/*.R directly and then cmd+shift+d / devtools::document() to regenerate
R/*.R files. Docs/fns in R/*.R is the new source of truth, and tools/make.R is just for
generating initial wrappers for new symbols")
# https://roxygen2.r-lib.org/articles/rd-formatting.html#code-chunks
# knit_print.python.builtin.object <- # strip env addressr
df |>
  group_by(file) |>
  dplyr::group_walk(\(df, grp) {

    txt <- df |>
      rowwise() |>
      mutate(final_dump = str_flatten_lines(
        # glue(r"--("{fs::path('man-src', r_name, ext = 'Rmd')}" # |>file.edit() # or cmd+click to edit man page)--"),
        # glue(r"--("{fs::path(man_src_dir, "0-upstream.md")}" # view the upstream doc)--"),
        # glue(r"--(#' @eval readLines("{fs::path(man_src_dir, "3-rendered.md")}") )--"),

        glue(r"--(#' {get_translated_lines(r_name)})--"),
        glue(r"--(#' @tether {endpoint})--"),
        str_c(r_name, " <- "),
        deparse(r_fn),
        ""
      )) |>
      _$final_dump

    txt <- str_flatten(
      c("## Autogenerated. Do not modify manually.", txt),
      "\n\n\n")

    txt <- txt |>
      str_split_lines() |>
      str_remove_non_ascii() |>
      str_trim("right") |>
      str_flatten_lines() |>
      str_trim()

    txt <- txt %>%
      str_flatten_and_compact_lines(roxygen = TRUE) %>%
      # handle empty @description TODO: handle this earlier, in dump.
      gsub("#' @description\n#'\n#' @", "#' @", ., fixed = TRUE)

    file <- paste0("R/autogen-", grp$file)
    writeLines(txt, file)
  })

if(FALSE) {

x <-  "abc\n@param foo baralkjasdf\n@param bar asdfklajfsa\n"
xx <- str_split_1(x, fixed("\n@"))
xx[startsWith(xx, "param ")] %<>%
  str_match_all("param ([^ ]+)(.*)") %>%
  map_chr(\(x) {
    stopifnot(nrow(x) == 1)
    name <- x[,2]
    desc <- glue::trim(str_trim(x[,3]))
    sprintf("param %s\n%s", name, desc)
  })

x2 <- xx %>%
  str_flatten("\n@") %>%
  str_replace_all("[\n]+@param", "\n\n@param") %>%
  str_replace_all("[\n]+@param", "\n\n@param")
cat(x2)
xx <- str_replace_all(x, "@param ([^ ]+)(.*)@", function(x){ browser()})
# xx <- str_match_all(x, regex("(.*)@param ([^ ]+)(.*)@", multiline = TRUE, dotall = T)) |> _[[1L]] #|> as.list()
xx <- str_locate_all(x, regex("@param ([^ ]+)(.*)@", multiline = TRUE, dotall = T))
xx
names(x) <- c("b", "name", "desc")
x$desc %<>% glue::trim()
# x <-
}


Sys.glob(c("man-src/*/2-*.Rmd")) %>%
  walk(\(f) {
    x <- read_lines(f)

    xx <- x |>
      # str_replace_all( "([^ ]+)\\$shape\\b", "shape(\\1)") |>
      str_trim("right")

    write_lines(xx, f)
  })

# dir_ls("man-src", glob = "1-*.md", recurse = TRUE) %>%
#
# Sys.glob(c("man-src/*/1-*.md", "man-src/*/2-*.Rmd")) %>%
#   walk(\(f) {
#     x <- read_file(f)
#
#     xx <- str_split_1(x, fixed("\n@"))
#     ip <- startsWith(xx, "param ")
#     xx[ip] <- xx[ip] %>%
#       map_chr(\(x) {
#         m <- str_match_all(x, regex("param ([^[:space:]]+)(.*)",
#                                     dotall = TRUE, multiline = TRUE))[[1]]
#         stopifnot(nrow(m) == 1)
#         # browser()
#         name <- m[,2]
#         desc <- glue::trim(str_trim(m[,3]))
#         sprintf("param %s\n%s\n", name, desc)
#       })
#
#     x2 <- xx %>%
#       str_flatten("\n@") %>%
#       str_replace_all("[\n]+@param", "\n\n@param") %>%
#       str_replace_all("[\n]+@param", "\n\n@param")
#
#     write_file(x2, f)
#   })
# man_src_pull_upstream_updates(write_out_only_formatted = TRUE)
# stop("FINITO")

man_src_pull_upstream_updates()


devtools::load_all() # TODO: render should be w/ an installed package and in a fresh r session w/ only `library(keras)`
man_src_render_translated()


envir::import_from(knitr, knit_print)
registerS3method("knit_print", "python.builtin.object", function(x, ...) {
  # browser()
  # utils::str(x)
  x <- capture.output(print(x))
  x <- trimws(x, "right")

  # strip object addresses; no noisy diff
  x <- sub(" at 0x[0-9A-Fa-f]{9}>$", ">", x, perl = TRUE)

  # remove reticulate hint from exceptions
  x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
  x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]
  writeLines(x)
})

process_chunk_output <- function(x, options) {
  # TKutils::str_vars(knitr::opts_knit$get("out.format"))

  message("process_chunk_output:")
  str(x)
  writeLines(x)
  cat("---\n")
  # utils::str(x)
  x_in <- x
  x <- x |> strsplit("\n") |> unlist() #|> trimws("right")
  x <- trimws(x, "right")

  # strip object addresses; no noisy diff
  x <- sub(" at 0x[0-9A-Fa-f]{9}>$", ">", x, perl = TRUE)

  # remove reticulate hint from exceptions
  x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
  x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]
  x <- paste0(x, collapse = "\n")
  if(x_in |> endsWith("\n") &&
     !x |> endsWith("\n"))
    x <- paste0(x, "\n")
  x
}

# we delay setting the output hook `knit_hooks$set(output = )` because
# if we set it too early, knitr doesn't set `render_markdown()` hooks.
# so we set a chunk option, which triggers setting the output hook
# one after knitr is already setup and knitting.
knitr::opts_hooks$set(
  keras.roxy.post_process_output = function(options) {
    message("Running option hook")
    str(options)

    # this is a self destructing option, run once before the first
    # chunk in a roxy block is evaluated
    options$keras.roxy.post_process <- NULL
    knitr::opts_chunk$set(keras.roxy.post_process = NULL)


    # make output reproducible
    # `evalenv` is created once per block, but knit() is called once per chunk
    # so we use this to detect if we're in the first chunk of a block and run setup
    if(is.null(roxygen2::roxy_meta_get("evalenv")$.__ran_block_init__)) {
      keras$utils$clear_session()
      set.seed(1L)
      keras$utils$set_random_seed(1L)
      assign(x = ".__ran_block_init__",
             envir = roxygen2::roxy_meta_get("evalenv"),
             value = TRUE)
    }

    local({

      og_output_hook <- knitr::knit_hooks$get("output")
      if(isTRUE(attr(og_output_hook, "keras.roxy.post_process", TRUE))) {
        message("Bailing early, not setting output hook")
        print(og_output_hook)
        return()
      }
      message("Setting output hook")
      knitr::knit_hooks$set(output = structure(function(x, options) {
        x <- process_chunk_output(x, options)
        og_output_hook(x, options)
      }, "keras.roxy.post_process" = TRUE))
    })
    options
  }
)

og_knit <- knitr::knit
unlockBinding("knit", asNamespace("knitr"))
knitr <- asNamespace("knitr")
knitr$knit <- function(input, output = NULL, tangle = FALSE, text = NULL,
                       quiet = FALSE, envir = parent.frame(), encoding = "UTF-8") {
  message("~~~~")
  message("Entering knit(), text = ")
  writeLines(c(text))
  ret <- og_knit(input, output, tangle, text,
          quiet, envir, encoding)
  message("Exiting knit(), ret =")
  writeLines(ret)
  message("_____")
  ret
}


options(warn = 1)
# trace(knitr::knit, quote())
devtools::document(roclets = c('rd', 'namespace', "roxytether::tether_roclet"))

# devtools::document(roclets = c('rd', 'namespace'))
envir::attach_source("tools/utils.R")
trimws_file("man/*.Rd")

# ok, so knit() is called once per chunk
# but the evalenv is created once per block
#
# we want to call clear_session() once per block
# so we need a way to detect from a chunk if we're in a block

# stop()
if(interactive()) local({
  rx <- callr::r_bg(\() remotes::install_local(force = TRUE, upgrade =  "never"))
  later::later(\() cat(rx$read_all_output()), delay = 17)
}) else {
  remotes::install_local(force = TRUE)
  rcmdcheck::rcmdcheck()
}


# remotes::update_packages(upgrade = "always")
# pkgdown::build_site()

# stop("DONE", call. = FALSE)
message("DONE!")

if(FALSE) {


}
