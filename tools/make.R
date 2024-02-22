#!/usr/bin/env Rscript

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")
# if(!"source:tools/translate-tools.R" %in% search()) envir::attach_source("tools/translate-tools.R")



## ---- Deferred ----

# TODO: k_fft() should really accept and return complex tensors too.
# TODO: activation_resolve() or activation_get() as alias of keras.activation.get() ?
# TODO: the 2-translated.Rmd should include a chunk w/ the function def (for easy seeing while editing)
#       with chunk options (include = FALSE)
# TODO: add PR for purrr::rate_throttle("3 per minute")
# TODO: layer_feature_space should take a formula, and dispatch to the features as required.
#       ~ scale(foo) * bar
# TODO: layer_feature_space() needs massaging.
# TODO: Add arg Layer(composing = TRUE)?
# TODO: custom keras.Model wrapper that patches add_weight(), build(), etc
#      with proper shape() coercion



## Rejected  ----
# TODO: k_array() should take a 'shape' argument
# TODO: r_name autogen: move "set" to tail, so have config_floatx(), config_floatx_set()
# TODO: implement dim() S3 generic. (will use shape() instead)


## Implemented but Questioning  ----


# TODO: pack_x_y_sample_weight() to return an unconverted py tuple?
#       Marked @internal for now
#       or just remove pack_x_y_sample_weight/unpack_x_y_sample_weight from the namespace?
#       No real utility in R since zeallot supports:
#         c(x, y = NULL, sample_weight = NULL) %<-% data

# TODO: remove usage of all.equal(<pyobj>, <pyobj>) in examples/code,
#       export a better way.
#       Exported all.equal S3 methods for KerasTensor and KerasVariable



## In process  ----


## next  ----

# TODO: R generator func should be passable to
#   adapt(layer_feature_space(), <r_generator>)


# TODO: op_vectorized_map() examples don't make sense

# TODO: note in docs for op_logical_and (and friends) that these are dispatched
#       to from & != and so on.

# TODO: this should work: op_convert_to_tensor(c(1, 3, 2, 0), "int32")

# TODO: op_arange: should it default to produce floats?

# TODO: "keras.applications.convnext" is a module, filtered out has good stuff

### internal utils ----
# TODO: as_0_based_index() utility: as_integer(x + 1)
# TODO: should as_axis() return a tuple() for lenght(x)>1 ?

# TODO: train_on_batch and related methods should be autogen'd and exported. Or maybe we curate those,
#       and don't export them? (I.e., have the few people that need them access methods via model$train_on_batch())

# TODO: model_from_saved_model() and model_to_saved_model(), provide guidance for users updating to 3.

# TODO: document get_initial_state() method for GRUCell and friends
#
# TODO: layer_lambda docs: bring back section on serialization and limitations after
#       fixing serialization.
#
# TODO: a layer_hashed_crossing() instance should have a customized wrapper that
#       splices in the args, so instead of layer(list(a, b)), you can do layer(a, b)
#       also, the example should maybe be nicer, with a named list/dict being passed,
#       instead of a tuple
#

# TODO: remove any tensorflow imports / DESCRIPTION deps



## Testing ----

# TODO: add tests for keras_history `history` from fit().
#   history - also mentions in docs (e.g., in callback_model_checkpoint())

# TODO: `backend()` used to have a `convert=FALSE` option, and all k_* would
#       return numpy arrays. We should check preserve np_array convert status in all
#       op_* functions, and return numpy arrays if we received a numpy array.
#       add tests for `use_backend("numpy")`

# TODO: add tests for R generator functions in fit()/evaluate()/predict().

# TODO: make sure tfdeploy::serve_savedmodel() works with the new
#       export_savedmodel() S3 method for keras3 models.
#       update example in export_savedmodel.keras.models.model.Model S3 method

# TODO: args 2 and 3 to op_svd() are seemingly broken/inconsistent across backends.
#
# TODO: op_istft op_irfft example is wrong, investigate
#

# TODO: feature_space saving errors

## Upstream bugs - add skipped tests
# TODO: BackupAndRestore is broken, doesn't respect current epoch. file/fix upstream.
#       ~/github/keras-team/keras/keras/callbacks/backup_and_restore_callback.py
# TODO: layer_torch_module_wrapper raises an error - maybe incompatible torch version?a
# TODO: this shouldn't error (empty last arg should be no-op):
# if(FALSE) {
# keras$utils$FeatureSpace$cross(
#   feature_names=c("string_values", "int_values"),
#   crossing_dim=32,
#   output_mode="int",
# )
# }
#
# TODO: layer_category_encoding()(count_weights) call arg example not working
# TODO: backout usage of `return_dict=TRUE` in evaluate() and friends - the output order is not stable.
#       use `setNames(as.list())`
#       ## Deferred until upstream bug fixed,
#       ## model.metrics_names returns wrong result

## Docs ----

### verbiage  ----

# TODO: global doc search for None/True/False/[Tt]uple/[Dd]ict(ionary|\\b) (Almost DONE except for Tuples)
# TODO: global search for "axis" in doc text, update to 1 based where appropriate.
# TODO: Clean up @family tags (partially autogen,
#       derive from module - then use that to autogenerate _pkgdown.yml)
# TODO: make the op_* @family tags make sense.
# TODO: @family tags should be manually managed, not dynamically generated.
#       perhaps in a yml file
# TODO: document options(keras.verbose) and other R global options.

# TODO: mention / document getOption(keras.*) #i.e., .fit_verbose, .verbose, etc.

# TODO: update application_preprocess_inputs man page: list the models for
#       which it's a "no-op" or "identity" function because the preprocessing
#       step is on the graph already.

# TODO: global search replace in man-src/*.Rmd "([^ ])=([^ ])" "\\1 = \\2"

# TODO: revisit k_vectorized_map() man page



### pipeline  ----

# TODO: fix links to chunk plots in rendered vignettes - should be relative to
#   package for R CMD build/pkgdown

# TODO: For apps, tether encode+decode along w/ constructor:
# #'  @tether application.foo,
# #'   application.foo.preprocess_input,
# #'   application.foo.decode_predictions,

# TODO: tether application process_utils

# TODO: many of the applications can share a man page,
#       e.g., application_convnext_{large...}

# TODO: refactor @tether so 'endpoint' can be a tuple like "keras.losses.Hinge,keras.losses.hinge"
#       and the tether file is a concatenation of multiple endpoints, separated
#       by a delimiter so we can do
#       read_file("0-upstream.md") |>
#         strsplit(str_c(strrep("~", 80), "\n")) |>
#         lapply(format_upstream)
# TODO: config_{enable,disable,is_enabled}_traceback_filtering have identical docstrings,
# should all be the same page (pending @tether improvements that allow for
#   multiple tethered endpoints.)



## Pending design decision

# [ method for tensors
#   make it 1 based?
#   update op_arange(), op_argmax(), op_argmin() with it to also be 1 based?

# TODO: how to pass layer call args through a wrapper that uses compose_layer()?

# TODO: op_array(<r_array>) should default to float32, not float64?

# TODO: layer_feature_space() needs many helpers for float_normalized() and friends
#       output_mode = 'dict' should be 'named list'?

# TODO: LossScaleOptimizer() default wrapper doesn't make sense - revisit

# TODO: fit/predict/eval/fit_on_batch... coerce `x` R arrays to:
#       float32?  model input$dtype?

# TODO: "keras.layers.InputSpec" - needs to be exported or somehow processed in `Layer()`

# TODO: export keras.DTypePolicy constructor - maybe module keras.dtype_policies

# TODO: ??? .onLoad(...) if(!interactive()) config_disable_interactive_logging()

# TODO: op_convert_to_numpy() -> rename to op_convert_to_array() or op_convert_to_r_array()
#       or just as.array()? remove it?
#       ?? rename op_array() to ... as_tensor() ?? ... or remove op_array()?

# TODO: get_custom_objects() needs thinking

# TODO: keras.Function ?? keras.Variable ?? keras.name_scope ??
# votes:
#   Yes to `keras_function()` (keras$Function)
#   Yes to `keras_tensor()` (keras$KerasTensor)
#   Document on same page for less confsion:
#      keras_input() adds batch dim to shape, keras_tensor() doesn't.
#   NO to keras$Variable()






## Pending need ----
## waiting for vignette-driven need

# TODO: custom_metric() / custom_loss() / Loss() / Metric(). remove custom_metric()?
# TODO: custom_metric -> metric_custom() / loss_custom() / constraint_custom()
#                        metric_lambda() / loss_lambda() / constraint_lambda()
#   waiting to see where else the need pops up, change maybe unnecessary

# TODO: CallbackList() (1 based) and MetricList() wrappers
# (and maybe convert LearningRateSchedule to 1 based?)
#  with offset occurring in the external py_to_r_wrapper (or simlar),
#  so that it presents in python code as 0 based, but presents in R
#  as a 1 based fn.

# TODO: revisit Model() signature, do we need update_state() and result() methods?

# TODO: what happened to is_keras_tensor() ??

# TODO: Model.get_compile_config() / Model.get_build_config() ?
#       Model.get_metrics_result() / Model.reset_metrics() /
#       Model.metrics / Model.metrics_names ?
#       Model.losses ?
#       Model.run_eagerly?
#       Model.stateless_call() ? Model.symbolic_call() ?

# TODO: fit()/ + friends, fix sample_weight / as_sample_weight,
#      - 1 based
#      - additional arg `sample_names`, to allow usage like
#        fit(..., class_weight = list(category_name_1 = .5, category_name_2 = .6),
#                 class_names = c("category_name_1", "category_name_2"))





# --- reticulate ----
# TODO: in reticulate: virtualenv_starter(): check for --enable-shared
# TODO: reticulate, support NO_COLOR (or similar) to disable the link
#        wrapper around `py_last_error()` hint.

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

if(FALSE) {


module_tether_blocks <- function(x) {
  lapply(unlist(x), \(name) glue::glue(r"--(
#' @title {name}
#' @name {name}
#' @tether {name}
#' @noRd
NULL

)--")) |> str_flatten("\n\n")
}

#' @title keras.ops
#' @name keras.ops
#' @tether keras.ops
#' @noRd
NULL

x <-
list_endpoints(include_modules = TRUE) %>%
  c("keras")
x %>%
  keep(\(e) inherits(py_eval(e), "python.builtin.module")) %>%
  grep("applications.", ., fixed = TRUE, invert = TRUE, value = TRUE) %>%
  grep("datasets.", ., fixed = TRUE, invert = TRUE, value = TRUE) %>%
  module_tether_blocks() |>
  cat_cb()

}


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

  "keras.dtype_policies",

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
  "keras.layers.Layer"             # only for subclassing  DONE
  "keras.callbacks.Callback"       # only for subclassing  DONE
  "keras.constraints.Constraint"   # only for subclassing  DONE
  "keras.losses.Loss"              # only for subclassing  DONE
  "keras.metrics.Metric"           # only for subclassing  DONE
  "keras.optimizers.Optimizer"     # only for subclassing
  "keras.regularizers.Regularizer" # only for subclassing
  "keras.initializers.Initializer" # only for subclassing
  "keras.optimizers.schedules.LearningRateSchedule"  # only for subclassing

  "keras.utils.PyDataset"      # parallel processing in R not possible this way
  "keras.utils.Sequence"       # tfdatasets is ~100x better anyway.

  "keras.utils.plot_model"        # S3 method plot()  DONE

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


# new symbols
df <- df %>%
  filter(!r_name %in% getNamespaceExports("keras3")) #%>% print(n = Inf)

intentionally_omited <- c(
  "callback_progbar_logger",
  "model_to_dot",
  "op_true_divide",
  "op_amax", "op_amin",
  "op_conjugate"
)

df <- df %>%
  filter(!r_name %in% intentionally_omited)

# print(df, n = Inf)



df %>%
  mutate(
    file = module %>%
      gsub("keras.src.", "", ., fixed = TRUE) %>%
      gsub(".", "-", ., fixed = TRUE) %>%
      sprintf("R/%s.R", .)
  ) %>%
  split_by(file) %>%
  walk(function(.df) {
    file = .df$file[1]

    dump <- .df$dump %>%
      paste0(collapse = "\n\n") %>%
      str_flatten_lines()

    cat(dump, file = file, append = TRUE)
  })

# stop("DONE")

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


devtools::load_all() # TODO: render should be w/ an installed package and in a fresh r session w/ only `library(keras3)`
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



## Waiting to be processed:

# Line wrapping for warnings is sorta broken, but can't seem to reprex.
# neither of these seem to reproduce the no-wrapping behavior for me...
# py_run_string(r"---(import os; os.system("for i in {1..1000}; do printf 'word '; done"))---")
# import warnings
# >>> warnings.warn("word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word")
# I believe this is a regression in the 2023.12.0 release of RStudio. Console output from reticulate / Python is no longer wrapped.
# This affects how Exceptions or Warnings from Python are presented (since they often do not include line breaks.)
# E.g, now I see something like there, where the warning is obscured
#
# Maybe there is some funky interaction w/ absl logging?
