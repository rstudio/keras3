#!/usr/bin/env Rscript


if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

# currently_accessed_keras_endpoints <-
#   list.files("R", pattern = "\\.R$", full.names = TRUE, all.files = TRUE) %>%
#   lapply(\(f) {
#     l <- readLines(f)
#     m <- str_extract_all(l, "keras\\$[a-zA-Z0-9$_.]+")
#     tibble(file = f, lines = l, matches = m, line_n = seq_along(l))
#   }) %>%
#   list_rbind()
#   unlist() %>% unique() %>%
#   lapply(\(x) {
#     list(endpoint = gsub("$", ".", x, fixed = TRUE),
#          py_obj = try(eval(str2lang(x)), silent = TRUE))
#   }) %>%
#   keep(\(x) inherits(x$py_obj, c("python.builtin.type", "python.builtin.function"))) %>%
#   list_transpose() %>%
#   as_tibble()

maybe_unbacktick <- function(x) {
  ticked <- startsWith(x, "`") & endsWith(x, "`")
  if(any(ticked))
    x[ticked] <- stringr::str_sub(x[ticked], 2L, -2L)
  x
}

df <-
  list.files("R", pattern = "\\.R$", full.names = TRUE, all.files = TRUE) %>%
  lapply(\(f) {
    l <- readLines(f)
    m <- str_extract_all(l, "`?keras\\$[a-zA-Z0-9$_.`]+")
    tibble(file = f, line = l, endpoint_expr = m, line_n = seq_along(l))
  }) %>%
  list_rbind() %>%
  tidyr::unchop(endpoint_expr) %>%
  rowwise() %>%
  mutate(
    py_obj = list(try(eval(str2lang(maybe_unbacktick(endpoint_expr))), silent = TRUE)),
    endpoint = endpoint_expr |>
      str_replace_all(fixed("$"), ".") |>
      str_replace_all(fixed("`"), "")
    ) %>%
  ungroup() %>%
  relocate(endpoint, py_obj, line, file)

accessed_endpoints <- df %>%
  filter(!startsWith(line, "#'")) %>%
  rowwise() %>%
  filter(inherits(py_obj, c("python.builtin.type",
                            "python.builtin.method",
                            "python.builtin.function"))) %>%
  ungroup()

# skipped <-
  df %>%
    anti_join(accessed_endpoints) %>%
    filter(!startsWith(line, "#'")) %>%
    arrange(map_chr(py_obj, \(x) class(x)[1]))


currently_accessed_keras_endpoints %>%
  rowwise() %>%
  filter(!inherits(py_obj, c("python.builtin.type",
                             "python.builtin.method",
                             "python.builtin.function"))) %>%
  ungroup() %>%
  filter(!line |> startsWith("#'")) %>%
  print(n = Inf)
# currently_defined_r_symbols <- callr::r(\() names(roxygen2::load_source(".")))
#
# find_endpoints_used <- function(expr) {
#   .recurse <- function(x) {
#     if(is.call(x) && identical(x[[1]], quote(`$`)) &&
#
#     return(.recurse(as.list(x)))
#
#   }
#   if(is.call(expr))
# }

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
