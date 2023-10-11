library(envir)
attach_source("tools/setup.R")
attach_source("tools/common.R")

# TODO: add PR for purrr::rate_throttle("3 per minute")

endpoints <- str_c("keras.", c(
  "activations",
  "applications",
  "regularizers",
  "callbacks",
  "constraints",
  "initializers",
  "datasets",
  "layers",
  "ops",
  # "losses",
  # "metrics",
  "optimizers"

  # "backend",
  # "dtensor",
  # "estimator",
  # "experimental", "export",
  # "Input",
  # "mixed_precision",
  # "Model",
  # "models",
  # "preprocessing",
  # "saving",
  # "Sequential",
  # "utils"
  )
  ) |> lapply(\(module) {
    nms <- names(py_eval(module)) |>
      setdiff(c("experimental", "deserialize", "serialize", "get"))
    str_c(module, nms, sep = ".")
    }) |>
  unlist() |> unique()

endpoints <-
  endpoints %>%
  tibble(endpoint = .) %>%
  mutate(py_obj = map(endpoint, py_eval),
         id = map_chr(py_obj, py_id),
         py_type = map_chr(py_obj, \(o) class(o) %>%
                             grep("^python\\.builtin\\.", ., value = TRUE) %>%
                             sub("python.builtin.", "", ., fixed = TRUE) %>%
                             .[1])) %>%
  split(., .$id) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # message(str_flatten_comma(df$endpoint))
    # if(any(grepl("AvgPool1D", df$endpoint))) browser()

    if(all(grepl(r"(keras\.layers\.(Global)?(Avg|Average|Max)Pool(ing)?[1234]D)",
                 df$endpoint))) {
      return(df %>% slice_max(nchar(endpoint)))
    } else {
      return(df %>% slice_min(nchar(endpoint), with_ties = FALSE))
    }
  }) %>%
  list_rbind() %>%
  split(., tolower(.$endpoint)) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # message(str_flatten_comma(df$endpoint))
    # prefer names w/ more capital letters
    df[which.max(nchar(gsub("[^A-Z0-9]*", "", df$endpoint))), ]
  }) %>%
  list_rbind() %>%
    # invisible()

  # aliases e.g., Conv2D, Convolution2D
  # summarize(endpoint = endpoint %>% .[which.min(nchar(.))]) %>%
  ungroup() %>%
  filter(map_lgl(endpoint, \(ep) inherits(py_eval(ep),
                                          c("python.builtin.type",
                                            "python.builtin.function")))) %>%
  .$endpoint %>%
  unlist() %>%
  # summarize()
  # grep("^keras\\.layers\\.GlobalAvgPool.D", ., value = TRUE, invert = TRUE) %>%  # alias for GlobalAveragePooling
  # grep("^keras\\.layers\\.GlobalMaxPool.D", ., value = TRUE, invert = TRUE) %>%  # alias for GlobalMaxPooling1D
  # grep("^keras\\.layers\\.[^.]*Convolution.D", ., value = TRUE, invert = TRUE) %>%  # alias for Conv1D
  # lapply(\(endpoint) {
  #   py_obj <- py_eval(endpoint)
  #   if(inherits(py_obj, c("python.builtin.type",
  #                         "python.builtin.function")))
  #     endpoint
  #   else NULL # filter out submodules, etc.
  #   ## rewrite this to be an rapply
  #   # "keras.applications.convnext" is a module, filtered out, has good stuff
  # }) |>
  unlist() %>%
  setdiff(c("keras.layers.Layer",
            "keras.layers.InputLayer",
            "keras.layers.InputSpec",
            "keras.optimizers.Optimizer",
            "keras.regularizers.Regularizer",
            "keras.constraints.Constraint"))

df <-
  endpoints |>
  # c( "keras.layers.InputSpec", "keras.layers.Input" ) |>
  map(mk_export) |>
  c(list(mk_layer_activation_selu())) |>
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

df |>
  mutate(endpoint_sans_name = str_extract(endpoint, "keras\\.(.*)\\.[^.]+$", 1)) |>
  filter(endpoint_sans_name %in% c("layers", "ops", "constraints",
                                   "activations", "regularizers")) |> #
  # select(endpoint, r_name, module, type) |>
  arrange(endpoint_sans_name, module, name) |>
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

    docstring <- map_chr(df$py_obj, \(p) {
      str_flatten_lines(
        str_c("# ", "keras$layers$", p$`__name__`),
        str_c("# ", p$`__module__`, ".", p$`__name__`),
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


#
#
# l <- keras$layers$PReLU
#
#
# l$`__doc__` |> trim() |> split_docstring_into_sections()
#
#   cat()
