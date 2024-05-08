
if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")
# if(!"source:tools/translate-tools.R" %in% search()) envir::attach_source("tools/translate-tools.R")



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
  # "keras.distribution",  # multi-host multi-device training
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
  grep("keras.distribution.", x = _, fixed = TRUE, value = TRUE) |>
  sort() |>
  purrr::set_names() |>
  lapply(mk_export)



df <- exports |>
  lapply(\(e) {
    unclass(e) |> map_if(\(attr) !is_scalar_atomic(attr), list) |>
      as_tibble_row()
  }) |>
  list_rbind() |>
  select(r_name, endpoint, type, module, everything())


cat(df$dump, file = "R/distribution.R", sep = "\n\n")
file.edit("R/distribution.R")

stop()

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
