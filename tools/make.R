library(envir)
attach_source("tools/setup.R")
attach_source("tools/common.R")

# TODO: add PR for purrr::rate_throttle("3 per minute")
#
# TODO: in reticulate: virtualenv_starter(): check for --enable-shared

# TODO: fix py_func(), for r_to_py.R6ClassGenerator
#   can't use __signature__ anymore in keras_core...
endpoints <- str_c("keras.", c %(% {

  "activations"  #
  "regularizers" #
  "callbacks"    #
  "constraints"  #
  "initializers" #
  "layers"       #
  "ops"          #
  "optimizers"   #
  "applications"
  "preprocessing"
  "preprocessing.image"
  "preprocessing.sequence"
  "losses"
  "metrics"
  "optimizers.schedules"

  # "saving"
  # "utils"
  # "backend"
  # "dtensor"
  # "mixed_precision"
  # "models"
  # "export"
  # "experimental"

  # "datasets"            #NO datasets unchanged, no need to autogen
  # "preprocessing.text"  #NO deprecated
  # "estimator"           #NO deprecated
}

  ) |> lapply(\(module) {
    message(module)
    module <- py_eval(modules <- module)
    nms <- names(module) |>
      setdiff(c("experimental", "deserialize", "serialize", "get",
                "keras_export",
                "ALL_OBJECTS", "ALL_OBJECTS_DICT"))
    nms <- gsub("`", "", nms)
    str_c(modules, nms, sep = ".")
    }) |>
  unlist() |> unique()

list_endpoints <- function(module = "keras", max_depth = 4,
                           skip = "keras.src", skip_regex = c(
                             "experimental", "deserialize", "serialize", "get"
                           )) {
  depth <- length(strsplit(module, ".", fixed = TRUE)[[1L]])
  if(depth > max_depth)
    return()
  module_py_obj <- py_eval(module)
  unlist(lapply(names(module_py_obj), \(nm) {
    endpoint <- paste0(module, ".", nm)
    if(endpoint %in% skip)
      return()
    # message(endpoint)
    endpoint_py_obj <- module_py_obj[[nm]]
    if(inherits(endpoint_py_obj, "python.builtin.module"))
      return(list_endpoints(endpoint))
    if(inherits(endpoint_py_obj, c("python.builtin.type",
                           "python.builtin.function")))
      return(endpoint)
    NULL
  }))
}

if(FALSE) {

  all_endpoints <- list_endpoints()

  all_endpoints %>%
    grep("ImageDataGenerator", ., value = T)

  all_endpoints %>%
    grep("Image", ., value = T, ignore.case = TRUE)

}

endpoints <-
  endpoints %>%
  tibble(endpoint = .) %>%
  mutate(py_obj = map(endpoint, py_eval)) %>%
  filter(map_lgl(py_obj, inherits, "python.builtin.object")) %>%
    mutate(
         id = map_chr(py_obj, py_id),
         py_type = map_chr(py_obj, \(o) class(o) %>%
                             grep("^python\\.builtin\\.", ., value = TRUE) %>%
                             sub("python.builtin.", "", ., fixed = TRUE) %>%
                             .[1])) %>%
  split(., .$id) %>%
  map(\(df) {
    # if(any(df$endpoint %>% grepl("FeatureSpace", .)))
    #   browser()
    if(nrow(df) == 1) return(df)
    # message(str_flatten_comma(df$endpoint))
    # if(any(grepl("AvgPool1D", df$endpoint))) browser()

    if(all(grepl(r"(keras\.layers\.(Global)?(Avg|Average|Max)Pool(ing)?[1234]D)",
                 df$endpoint))) {
      return(df %>% slice_max(nchar(endpoint)))

    } else
    #   if (any(str_detect(df$endpoint, "metrics|losses"))) {
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
    # } else
      {

      return(df %>% slice_min(nchar(endpoint), with_ties = FALSE))
    }
  }) %>%
  list_rbind() %>%
  split(., tolower(.$endpoint)) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # prefer names w/ more capital letters
    df[which.max(nchar(gsub("[^A-Z0-9]*", "", df$endpoint))), ]
  }) %>%
  list_rbind() %>%

  # aliases e.g., Conv2D, Convolution2D
  # summarize(endpoint = endpoint %>% .[which.min(nchar(.))]) %>%
  ungroup() %>%
  filter(map_lgl(endpoint, \(ep) inherits(py_eval(ep),
                                          c("python.builtin.type",
                                            "python.builtin.function")))) %>%
  .$endpoint %>%
  unlist() %>%
  #   # "keras.applications.convnext" is a module, filtered out, has good stuff
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

    "keras.metrics.Accuracy"         # weird, only class handle, no fn handle - weird alias
                                     # for binary_accuracy, but without any threshold casting.
                                     # kind of confusing - the keras.metrices.<type>*_accuracy
                                     # endpoints are much preferable.

    "keras.layers.Wrapper"           # needs thinking
    "keras.layers.InputLayer"        # use Input instead
    "keras.layers.InputSpec"         # ??
    "keras.callbacks.CallbackList"   # just an abstract list
    "keras.callbacks.History"        # always added to by default

    "keras.constraints.to_snake_case" # ??
    "keras.optimizers.LegacyOptimizerWarning"
  })

# filter out function handles that also have class handles
endpoints <- endpoints %>%
  lapply(\(endpoint) {
    if(!any(startsWith(endpoint, c("keras.losses.", "keras.metrics."))))
      return(endpoint)

    py_obj <- py_eval(endpoint)
    if (!inherits(py_obj, "python.builtin.function"))
      return(endpoint)

      # if we have both functional and type api interfaces, we
      # just want the type right now (we dynamically fetch the matching
      # functional counterpart later)
      class_endpoint <- switch(endpoint,
             "keras.losses.kl_divergence" = "keras.losses.KLDivergence",
             str_replace(endpoint, py_obj$`__name__`,
                         snakecase::to_upper_camel_case(py_obj$`__name__`)))
      tryCatch({
        py_eval(class_endpoint)
        return(NULL)
      },
      python.builtin.AttributeError = function(e) {
        # don't emit warning about known function handles without class handle
        # counterparts
        if (!endpoint %in% sprintf(
          "keras.metrics.%s", c(
            "binary_focal_crossentropy",
            'categorical_focal_crossentropy',
            'huber',
            'kl_divergence',
            'log_cosh')))
          # browser()
          print(e)
        endpoint
      })
  }) %>%
  unlist() %>%
  invisible()

get_metric_counterpart_endpoint <- function(..., fn, type) {
  if(missing(fn)) {
    endpoint <- type
    py_obj <- py_eval(endpoint)
    name <- py_obj$`__name__`
    endpoint_fn <- str_replace(endpoint, name, snakecase::to_snake_case(name))
    py_obj_fn <- py_eval(endpoint_fn) %error% browser()
  }
}

# TODO: initializer families:
# <class 'keras.initializers.constant_initializers.Zeros'>
# <class 'keras.initializers.random_initializers.RandomUniform'>

# TODO: next: losses, metrics, saving, guides/vignettes

df <-
  endpoints |>
  lapply(\(e) {
    # message(e)
    mk_export(e)
  }) |>
  # c(list(mk_layer_activation_selu())) |>
  map(\(e) {
    # if(e$endpoint == "keras.activations.selu")
    #   browser()
    e |>
      unclass() |>
      map_if(\(attr) ! is_scalar_atomic(attr), list) |> #str()
      as_tibble_row()
  }) |>
  list_rbind()

# TODO: bidirectional, time_distributed -- need special caseing

df <- df |>
  mutate(endpoint_sans_name = str_extract(endpoint, "keras\\.(.*)\\.[^.]+$", 1))

df |>
  filter(endpoint_sans_name %in% c("layers", "ops", "constraints", "initializers",
                                   "callbacks", "optimizers",
                                   "preprocessing",
                                   "preprocessing.image",
                                   "preprocessing.sequence",
                                   "losses", "metrics",
                                   "optimizers.schedules",
                                   # "utils",
                                   "applications",
                                   "activations", "regularizers")) |> #
  # select(endpoint, r_name, module, type) |>
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
                        str_c(endpoint_sans_name, ".R")
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

    df$module %<>% str_replace_all(fixed("keras."), "keras_core.")
    docstring <- map_chr(df$py_obj, \(p) {
      str_flatten_lines(
        str_c("# ", "keras$layers$", p$`__name__`),
        str_c("# ", p$`__module__`, ".", p$`__name__`) |> str_replace_all(fixed("keras."), "keras_core."),
        str_flatten(c('r"-(', p$`__doc__`, ')-"'), "")
      )
    })
    df$dump %<>% str_c("# ", df$module, ".", df$name, "\n", .)

    txt <- c("## Autogenerated. Do not modify manually.",
             str_c(docstring, "\n\n\n", df$dump)) %>%
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



#
#
# l <- keras$layers$PReLU
#
#
# l$`__doc__` |> trim() |> split_docstring_into_sections()
#
#   cat()
