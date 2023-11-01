
# if(!"source:tools/utils.R" %in% search())
  # envir::attach_source("tools/utils.R")
source("tools/utils.R")

# TODO: add PR for purrr::rate_throttle("3 per minute")
#
# TODO: in reticulate: virtualenv_starter(): check for --enable-shared

# TODO: fix py_func(), for r_to_py.R6ClassGenerator
#   can't use __signature__ anymore in keras_core...

## TODO: "keras.applications.convnext" is a module, filtered out, has good stuff

# TODO: initializer families:
# <class 'keras.initializers.constant_initializers.Zeros'>
# <class 'keras.initializers.random_initializers.RandomUniform'>

# TODO: next: losses, metrics, saving, guides/vignettes

# TODO: bidirectional, time_distributed -- need special caseing
#
# TODO: note in docs for k_logical_and (and friends) that these are dispatched
#       to from & != and so on.

# TODO: keras.Function ?? keras.Variable ?? keras.name_scope ??
#
# TODO: remove k_random_binomial() ??
#
# TODO: layer_feature_space() needs massaging.
#
# TODO: to_categorical():
#    - handle factor/character https://github.com/rstudio/keras/issues/1055
#    - make it 1 based?

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

  "keras.datasets",            # datasets unchanged, no need to autogen
  "keras.preprocessing.text",  # deprecated
  "keras.estimator",           # deprecated
  "keras.optimizers.legacy",

  "keras.src"                  # internal
)) %>%  c(
  list_endpoints("keras.applications", max_depth = 0)) %>%
  # filter out top level non module symbols for now
  grep("keras.([^.]+)$", ., value = TRUE, invert = TRUE) %>%
  unique()


# some endpoints are aliases. resolve unique endoints.
endpoints <-
  endpoints %>%
  tibble(endpoint = .) %>%
  mutate(py_obj = map(endpoint, py_eval)) %>%

  # filter non py objects, e.g., version strings
  filter(map_lgl(py_obj, inherits, "python.builtin.object")) %>%

  mutate(
    id = map_chr(py_obj, py_id),
    py_type = map_chr(py_obj, \(o) class(o) %>%
                        grep("^python\\.builtin\\.", ., value = TRUE) %>%
                        sub("python.builtin.", "", ., fixed = TRUE) %>%
                        .[1])) %>%

  # filter out literal aliases, i.e., identical py ids.
  dplyr::group_split(id) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # for the pooling layer aliases, pick the longer/clearer name
    if(all(grepl(r"(keras\.layers\.(Global)?(Avg|Average|Max)Pool(ing)?[1234]D)",
                 df$endpoint)))
      return(df |> slice_max(nchar(endpoint)))

    # keep aliases of losses under metrics, for complete autocomplete with 'metric_'
    if(all(df$endpoint |> str_detect("^keras\\.(metrics|losses)\\."))) {
      # message("keeping aliases: ", str_flatten_comma(df$endpoint))
      return(df)
    }



    # if(any(df$endpoint %>% str_detect("keras.ops.average_pool"))) browser()
    return(df %>% filter(py_obj[[1]]$`_api_export_path`[[1]] == endpoint))


    # otherwise, default to picking the shortest name, but most precise
    # sort keras.preprocessing before keras.utils
    df |>
      mutate(
        name = str_split(endpoint, fixed(".")) %>% map_chr(., ~.x[length(.x)]),
        submodule = str_split(endpoint, fixed(".")) %>% map_chr(., ~.x[length(.x)-1])
      ) |>
      arrange(nchar(name), submodule) |>
      slice(1)
      # slice_min(nchar(endpoint), n = 1, with_ties = FALSE)
  }) %>%
  list_rbind() %>%

  # filter duplicate endpoint names, where all that differs is capitalization.
  # this mostly affects endpoints offered as both function and class handles:
  # metrics, losses, and merging layers.
  # if we have both functional and type api interfaces, we
  # just want the type right now (we dynamically fetch the matching
  # functional counterpart later as needed, e.g., metrics)
  split(., snakecase::to_upper_camel_case(.$endpoint)) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # prefer names w/ more capital letters (i.e., the class handles)
    i <- which.max(nchar(gsub("[^A-Z]*", "", df$endpoint)))
    # message(sprintf("keep: %s drop: %s",
    #                 str_pad(df$endpoint[i], 50, "right"),
    #                 str_flatten_comma(df$endpoint[-i])))
    df[i,]
  }) %>%
  list_rbind() %>%

  .$endpoint %>% unlist() %>%

  ## filter out some endpoints that need special handling
  setdiff(c %(% {
    "keras.layers.Layer"             # only for subclassing
    "keras.optimizers.Optimizer"     # only for subclassing
    "keras.regularizers.Regularizer" # only for subclassing
    "keras.constraints.Constraint"   # only for subclassing
    "keras.initializers.Initializer" # only for subclassing
    "keras.callbacks.Callback"       # only for subclassing
    "keras.losses.Loss"              # only for subclassing
    "keras.metrics.Metric"           # only for subclassing
    "keras.optimizers.schedules.LearningRateSchedule"  # only for subclassing

    "keras.utils.PyDataset"
    "keras.utils.Sequence"           # parallel processing in R no possible this way
                                     # tfdatasets is ~100x better anyway.

    "keras.utils.plot_model"        # S3 method plot()

                                       # TODO: revisit custom_object_scope()
    "keras.utils.custom_object_scope"  # need custom work to resolve py_names -
                                       # manually wrapped to `with_custom_object_scope()`

    "keras.metrics.Accuracy"         # weird, only class handle, no fn handle - weird alias
                                     # for binary_accuracy, but without any threshold casting.
                                     # kind of confusing - the keras.metrices.<type>*_accuracy
                                     # endpoints are much preferable.

    "keras.utils.Progbar"           # needs thinking
    "keras.layers.Wrapper"           # needs thinking
    "keras.layers.InputLayer"        # use Input instead
    "keras.layers.InputSpec"         # ??
    "keras.callbacks.CallbackList"   # just an abstract list
    "keras.callbacks.History"        # always added to fit() by default

    "keras.optimizers.LegacyOptimizerWarning"
  })


df <- endpoints |> purrr::set_names() |>
  lapply(mk_export) |>
  lapply(\(e) {
    e |>
      unclass() |>
      map_if(\(attr) ! is_scalar_atomic(attr), list) |>
      as_tibble_row()
  }) |>
  list_rbind()

df <- df |>
  mutate(endpoint_sans_name = str_extract(endpoint, "keras\\.(.*)\\.[^.]+$", 1))

df |>
  mutate(endpoint_sans_name = endpoint_sans_name %>%
           replace_val(c("preprocessing.image", "preprocessing.sequence"),
                       "preprocessing")) %>%
  arrange(endpoint_sans_name, module, r_name) |>
  mutate(file = if_else(endpoint_sans_name == "layers",
                        {
                          # browser()
                        module |>
                          str_replace("^keras(_core)?\\.(src\\.)?", "") |>
                          str_replace(paste0(endpoint_sans_name, "\\."), "") |>
                          str_replace("^([^.]+).*", paste0(endpoint_sans_name, "-\\1.R")) #|> unique()
                          # str_sub("\\.?src\\.?", "") |>
                          # str_sub(fixed("."), "-") |>
                          # str_c(".R")
                        },
                        str_c(endpoint_sans_name %>% str_replace_all(fixed("."), "-"), ".R")
                        )
                        ) |>
  # select(endpoint, endpoint_sans_name, module, file)  |> print(n=Inf)
  group_by(file) |>
  dplyr::group_walk(\(df, grp) {

    # o <- df$py_obj
    # docstring <- str_c("# ", "keras$layers$", l$`__name__`),
    # str_c("# ", l$`__module__`, ".", l$`__name__`),
    # str_flatten(c('r"-(', l$`__doc__`, ')-"'), ""),
    # "\n",

#
#     fs::dir_create("tools/raw")
#     withr::with_dir("tools/raw", {
#       endpoint_dir <-df$endpoint |>
#         str_replace_all(fixed("."), "-") |>
#         str_replace_all(fixed("^keras-"), "")
#       fs::dir_create(endpoint_dir)
#       mapply(writeLines, df$dump, file.path(endpoint_dir, "r-wrapper-literal.R"))
#       mapply(writeLines, df$dump, file.path(endpoint_dir, "r-wrapper-llm.R"))
#       mapply(writeLines, trim(df$docstring), file.path(endpoint_dir, "docstring.R"))
#     })

    docstring <- map2_chr(df$py_obj, df$endpoint, \(p, ep) {
      str_flatten_lines(
        # str_c("# ", deparse1(endpoint_to_expr(ep))),
        str_c("# ", ep),
        str_c("# ", p$`__module__`, ".", p$`__name__`), # |> str_replace_all(fixed("keras."), "keras_core."),
        str_flatten(c('r"-(', p$`__doc__`, ')-"'), "")
      )
    })


    # df$dump %<>% str_c("# ", df$module, ".", df$name, "\n", .)

    txt <- c("## Autogenerated. Do not modify manually.",
             str_c(docstring, "\n", df$dump)) %>%
      str_flatten("\n\n\n") %>% {
        while (nchar(.) != nchar(. <- gsub("#'\n#'\n", "#'\n", ., fixed = TRUE))) {} # TODO: do this in dump
        while (nchar(.) != nchar(. <- gsub("\n\n\n\n", "\n\n\n", ., fixed = TRUE))) {}
        .
      } %>%
      # handle empty @description TODO: handle this earlier, in dump.
     gsub("#' @description\n#'\n#' @", "#' @", ., fixed = TRUE)


    file <- paste0("R/autogen-", grp$file)
    # if (grp$endpoint_sans_name == "layers") {
    #   file <- "R/layers.R"
    #   unlink(file)
    # }
    # else
    #   file <- glue("R/autogen-{grp$endpoint_sans_name}.R")

    writeLines(txt, file)
  })

# devtools::document()

stop("DONE", call. = FALSE)
# filter(endpoint_sans_name %in% c("layers", "ops", "constraints", "initializers",
#                                  "callbacks", "optimizers",
#                                  "preprocessing",
#                                  "preprocessing.image",
#                                  "preprocessing.sequence",
#                                  "losses", "metrics",
#                                  "optimizers.schedules",
#                                  # "utils",
#                                  "applications",
#                                  "activations", "regularizers")) |> #
# select(endpoint, r_name, module, type) |>
# |>
  dplyr::bind_rows()

map(xx, format)

  # tibble(endpoint = _) |>
  mutate(name = endpoint |>
           str_replace_all(fixed("."), "$") |>
           map(\(s) names(eval(str2lang(s))))) |>
  tidyr::unchop(name) |>
  filter(!name %in% c("experimental", "deserialize", "serialize")) |>
  mutate(endpoint = str_c(endpoint, name, sep = ".")) |>
  filter(!endpoint %in% "keras.Sequential.build") #|>


# k_fft_2 vs k_fft2
# start w/ the export endpoint, and augment it.



x <- mk_export("keras.layers.PReLU")
x


# TODO:
#
# Parsing of params for keras$ops$tensordot isn't right
#
# TODO: @returns required for every obj
#
#
# TODO:
# layer_wrapper?
# layer_torch_module_wrapper?
# layer_time_distributed?
# layer_tfsm_layer?
# layer_bidirectional

# Seemingly removed:
# layer_random_height
# layer_random_width
# layer_locally_connected_{12}d
# layer_cudnn_*
# layer_dense_features
# layer_alpha_dropout
# layer_activation_thresholded_relu

make_r_fn("keras.layers.PReLU")

mk_export("keras.layers.PReLU")
mk_export("keras.applications.VGG16")


mk_export("keras.ops.conv_transpose")$dump |> cat()

roxygen2::parse_file("/Users/tomasz/github/rstudio/keras/R/autogen-layers-core.R") -> r

# TODO: UpSampling1D needs fixup

r[[3]] -> r

r
r$tags[[2]]
trace(system2, quote(message(paste("+", paste0(env, collapse = " "),
                                   shQuote(command), paste0(shQuote(args), collapse = " "))))); tools:::..Rd2pdf(".")


mk_export("keras.layers.LayerNormalization")$doc$description -> d


str()

  str_sppulit_1(fixed("\n```")) %>%
  {
    length(.) <- ceiling(length(.)/2)*2 # maybe resize to multiple of 2
    dim(.) <- c(2, length(.) / 2)
    rownames(.) <- c("prose", "code")
    .
  } %>% {
    p <- .["prose",]

  }
  print()
  str()


# TODO: callback_progbar logger ... the documented arg doesn't seem to actually be accepted,
# also, does it make sense to export it?
#
  "keras.constraints.to_snake_case" # ??

  # if (any(str_detect(df$endpoint, "metrics|losses"))) {
  #
  #   nms_in <- names(df)
  #   df <- df |>
  #     mutate(
  #       name = map_chr(py_obj, \(o) o$`__name__`),
  #       module = map_chr(py_obj, \(o) o$`__module__`)) |>
  #     rowwise() %>%
  #     mutate(
  #       dist_from_true_home = adist(str_replace(endpoint, name, ""),
  #                                   module)
  #     ) |>
  #     ungroup() |>
  #     arrange(dist_from_true_home, desc(nchar(name)))
  #     # message(str_flatten_comma(df$endpoint, ", "))
  #   df %>%
  #     # slice(1) %>%
  #     select(!!nms_in)

  # } else {

#
#
# l <- keras$layers$PReLU
#
#
# l$`__doc__` |> trim() |> split_docstring_into_sections()
#
#   cat()
