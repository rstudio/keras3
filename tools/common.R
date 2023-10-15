

# ---- docstrings ----

known_section_headings <- c(
  "Args:",
  "Arguments:",
  "Attributes:",
  "Returns:",
  "Raises:",
  "Input shape:",
  "Inputs:",
  # "Input:",
  "Output shape:",
  # "Outputs:",
  "Output:",
  "Call arguments:",
  "Call Args:",
  "Returns:",
  "Example:",
  "References:",
  "Reference:",
  "Examples:"
)


# PROMPT: keyword arguments -> named list
# PROMPT: list/tuple -> list
# PROMPT: None -> NULL
# PROMPT: True -> TRUE
# PROMPT: False -> FALSE


# TODO: @references tag for "References:" section
tidy_section_headings <- known_section_headings |>
  str_sub(end = -2) |>
  replace_val("Args", "Arguments") |>
  replace_val("Example", "Examples") |>
  replace_val("Call Args", "Call arguments") |>
  # replace_val("Inputs", "Input shape") |>
  # replace_val("Outputs", "Output shape") |>
  snakecase::to_snake_case()



split_docstring_into_sections <- function(docstring) {

  if(is.null(docstring))
    return(NULL)
  assert_that(docstring |> is_string()) %error% browser()

  docstring <- docstring |>
    str_split_lines() |> str_trim("right") |> str_flatten_lines()

  docstring <- docstring |> fence_in_examples_w_prompt_prefix()

  for(h in known_section_headings) {
    # "Input shape:" not always on own line :*-(
    docstring <- gsub(paste0(h, " "), paste0("\n", h, "\n"), docstring, fixed = TRUE)
  }

  docstring <- docstring %>%
    str_split_lines() %>%
    # Input shape section in keras.layers.ConvLSTM{12}D has badly formatted item list
    # (indentation)(wtf) - (If)  -> (indentation)(wtf)\n(indentation) - (If)
    sub("^([ ]*)(.*[^ ].*)+[ ]+- (If|Else)", "\\1\\2\n\\1- \\3", .) %>%
    str_flatten_lines()

  x <- str_split_1(docstring, "\n")
  if(length(x) <= 3)
    return(list(title = x[1],
                description = str_flatten(x[-1], "\n")))

  m <- match(str_trim(x), known_section_headings)
  m <- zoo::na.locf0(m)
  m <- tidy_section_headings[m]

  m[1:2] <- m[1:2] %|% c("title", "description")
  m <- zoo::na.locf0(m)

  # if(grepl("BatchNormalization", docstring))
  #   browser()

  # TODO: in BatchNormalization, there is more prose after "Reference:" section
  ind_lvl <- str_width(str_extract(x, "^[ ]*"))
  maybe_new_sec <- which(m == "reference" &
                       ind_lvl == 0 &
                       str_width(str_trim(x)) > 0 &
                       !startsWith(str_trim(x), "- "))[-1]
  if(length(maybe_new_sec)) {
    m[maybe_new_sec[1]:length(m)] %<>% replace_val("reference", "description")
    append(m, after = maybe_new_sec[1]) <- "description"

    x[maybe_new_sec[1]] %<>% paste0("\n", .)
    x %<>% str_split_lines()
  }

  sections <- split(x, m) |>
    imap(\(s, nm) {
      if (!nm %in% c("description", "title", "details", "note"))
        s <- s[-1] # drop heading
      s |> str_trim("right") |> str_flatten("\n") |> trim()
    })

  sections[unique(c(m))] # split() reorders
}


parse_params_section <- function(docstring_args_section) {

  for (chk_empty_fn in c(is.null, is.na, partial(`==`, "")))
    if(chk_empty_fn(docstring_args_section))
      return(list())

  if(!is_scalar(docstring_args_section))
    docstring_args_section %<>% str_flatten("\n")


  x <- docstring_args_section |>
    c("") |> # append final new line to ensure glue::trim() still works when length(x) == 1
    str_flatten("\n") |> glue::trim() |> str_split_1("\n")

  m <- str_match(x, "^\\s{0,3}(?<name>[*]{0,2}[A-Za-z0-9_]+) ?: (?<desc>.+)$")

  description <- m[,"desc"] %|% x
  param_name <- m[,"name"]

  not_param <- which(param_name == "Default")
  if(length(not_param)) {
    description[not_param] <- x[not_param]
    param_name[not_param] <- NA
  }

  param_name[startsWith(param_name, "**")] <- "..."
  param_name[param_name == "kwargs"] <- "..."

  param_name %<>% zoo::na.locf0()

  {if(any(startsWith(param_name, "*"))) browser()} %error% browser()
  # if both *args and **kwargs are documented, collapse them into the "..." param (todo)

  out <- split(description, param_name) |>
    map(\(s) s|> str_trim("right") |> str_flatten("\n"))

  out <- out[unique(param_name)] # preserve original order
  out
}

# TODO: bring back callback_backup_and_restore()?

fence_in_examples_w_prompt_prefix <- function(docstring) {

  # fence in code examples that are missing fences
  # but start with >>>
  docstring <- str_split_1(docstring, "\n") |> str_trim("right")
  docstring0 <- str_trim(docstring)
  in_code_block <- FALSE
  for (i in seq_along(docstring)) {

    if (!in_code_block) {
      # not currently in a code block, look for the start of a code block
      if (startsWith(docstring0[[i]], ">>>")) {
        in_code_block <- TRUE

        docstring[[i]] <- str_c("```python\n",
                                str_replace(docstring[[i]], ">>> ", ""))
      }

    } else {
      # code block context open,

      # look for the end of the code block
      if (docstring0[[i]] == "") {
        docstring[[i]] <- "```\n"
        in_code_block <- FALSE

        # tidy up example commands
      } else if(any(startsWith(docstring0[[i]], c(">>> ", "... ")))) {
        docstring[[i]] <- str_sub(docstring[[i]], 5L)
      } else
        # tidy up example output
        docstring[[i]] <- str_c("# ", docstring[[i]])
    }
  }
  if(in_code_block)
    docstring <- c(docstring, "```\n")

  docstring <- str_flatten(docstring, collapse = "\n")

}


# ---- roxygen -----

make_roxygen_tags <- function(endpoint, py_obj, type) {
  if(type == "layer")
    family <- get_layer_family(py_obj)
  else if (endpoint |> startsWith("keras.ops."))
    family <- "ops"
  else if (endpoint |> startsWith("keras.activation"))
    family <- "activation functions"
  else
    family <- type

  out <- list()
  out$export <- ""

  # family is.na for dense_features
  if (isTRUE(family != ""))
    out$family <- family

  link <- get_tf_doc_link(endpoint)
  out$seealso <- sprintf("\n+ <%s>", link)

  out
}


get_tf_doc_link <- function(endpoint) {
  url_tail <- str_replace_all(endpoint, fixed('.'), '/')
  glue("https://www.tensorflow.org/api_docs/python/tf/{url_tail}")
}


get_layer_family <- function(layer) {

  family <- layer$`__module__` |>
    str_extract(".*\\.layers\\.(.*)s?\\.", group = 1) |>
    replace_val("rnn", "recurrent") |>
    str_replace_all("_", " ")


  if(is.na(family) && py_is(layer, keras$layers$Input))
    family <- "core"

  str_c(family, " layers")
}


# ---- r wrapper --------

make_r_name <- function(endpoint, module = py_eval(endpoint)$`__module__`) {

  if(endpoint |> startsWith("keras.ops."))
    endpoint %<>% str_replace(fixed("keras.ops."), "keras.k.")
  x <- str_split_1(endpoint, fixed("."))

  name <- x[length(x)] # __name__
  x <- x[-length(x)] # submodules

  if(length(x) >= 2)
    x <- x[-1] # drop "keras" from "keras.layers.Dense
  stopifnot(is_scalar(x))
  prefix <- x |> str_replace("s$", "") |> str_flatten("_")

  name <- name |>
    str_replace("NaN", "Nan") |>
    str_replace("RMSprop$", "Rmsprop") |>
    str_replace("ReLU$", "Relu") |>
    str_replace("ResNet", "Resnet") |>
    str_replace("ConvNeXt", "Convnext") |>
    str_replace("XLarge", "Xlarge") |>
    # str_replace("EfficientNet", "Efficientnet") |>

    snakecase::to_snake_case() |>

    str_replace("_([0-9])_d(_|$)", "_\\1d\\2") |>  # conv_1_d  ->  conv_1d
    # str_replace("efficient_net_(.+)$", "efficientnet_\\1") |>

    # str_replace("re_lu", "relu") |>
    str_replace_all("(^|_)l_([12])", "\\1l\\2") |> # l_1_l_2 -> l1_l2

    # applications
    str_replace_all("_v_([1234])($|_)", "_v\\1\\2") |>
    str_replace_all("^resnet_([0-9]+)", "resnet\\1") |>
    str_replace_all("^vgg_([0-9]+)", "vgg\\1") |>
    str_replace_all("^dense_net_", "densenet") |>
    str_replace_all("^mobile_net?", "mobilenet") |>



    str_replace_all("max_norm", "maxnorm") |>
    str_replace_all("non_neg", "nonneg") |>
    str_replace_all("min_max", "minmax") |>
    str_replace_all("unit_norm$", "unitnorm") |>
    str_replace("tensor_board", "tensorboard") |>
    identity()

  if (str_detect(name, "efficient_net_"))
    name <- name |>
    str_split_1("efficient_net_") |> str_replace_all("_", "") |>
    str_flatten("efficientnet_")

  if(str_detect(name, "nas_net_"))
    name %<>% str_replace_all("_", "")


  if(prefix != "k")
    name %<>% str_replace(glue("_{prefix}$"), "")
  name <- str_c(prefix, "_", name)

  ## fixes for specific wrappers
  # no _ in up_sampling
  name %<>% str_replace("^layer_up_sampling_", "layer_upsampling_")

  # activation layers get a prefix (except for layer_activation())
  if(startsWith(name, "layer_") &&
     str_detect(module, "\\.activations?\\.") &&
     name != "layer_activation")
    name %<>% str_replace("^layer_", "layer_activation_")

  name <- name |>
    replace_val("layer_activation_p_relu", "layer_activation_parametric_relu") |>
    replace_val("regularizer_orthogonal_regularizer", "regularizer_orthogonal") |>
    # replace_val("callback_lambda_callback", "callback_lambda") |>
    # replace_val("callback_callback_list", "callback_list") |>
    identity()


  name
}

#TODO: param parsing in AdamW borked
#TODO: revisit application helpers like xception_preprocess_input()
#TODO: KerasCallback and other R6 classes for subclassing...
#TODO: implementation() - fix up docs / actual

transformers_registry <-
  yaml::read_yaml("tools/arg-transformers.yml") %>%
  imap(\(args, endpoint) {
    if(!str_detect(endpoint, "[*?]")) # not a glob
      names(args) <- str_c("+", names(args))
    lapply(args, function(fn) {
      fn <- str2lang(fn)
      if (is.call(fn) && !identical(fn[[1]], quote(`function`)))
        fn <- as.function.default(c(alist(x =), fn))
      fn
    })
  })


get_arg_transformers <- function(endpoint, py_obj = py_eval(endpoint), params = NULL) {

  if(is.null(params)) {
    params <-  py_obj$`__doc__` |> trim() |>
      split_docstring_into_sections() |>
      _$arguments |> parse_params_section()
  }

  transformers <- list()
  frmls <- formals(py_obj)

  pre_registered <- transformers_registry %>%
    .[str_detect(endpoint, glob2rx(names(.)))] %>%
    unname() %>% unlist(recursive = FALSE, use.names = TRUE)

  add_if_present     <- pre_registered %>% .[!startsWith(names(.), "+")]
  add_always         <- pre_registered %>% .[startsWith(names(.), "+")]
  names(add_always) %<>% str_sub(2, -1)
  rm(pre_registered)

  # build up a default list
  #  - integer default values get a "as_integer"
  #  - "int" in param description gets a "as_integer"
  for (key in c(names(frmls), names(params))) {

    if (!is.null(tr <- add_if_present[[key]])) {
      transformers[[key]] <- tr
    } else if (typeof(frmls[[key]]) == "integer" ||
               str_detect(params[[key]] %||% "",
                          regex("\\bint(eger)?\\b", ignore_case = TRUE))) {
      transformers[[key]] <- quote(as_integer)
    }

  }

  transformers %<>% modifyList(add_always)

  if (!length(transformers))
    transformers <- NULL

  transformers
}


make_r_fn <- function(endpoint,
                      py_obj = py_eval(endpoint),
                      type = keras_class_type(py_obj),
                      transformers = get_arg_transformers(endpoint, py_obj, params = NULL)) {

  # TODO: rename keras_class_type to keras_endpoint
  if(endpoint |> startsWith("keras.ops.")) {
    return(make_r_fn.op(endpoint, py_obj, transformers))
  }
  if(endpoint |> startsWith("keras.activation")) {
    return(make_r_fn.activation(endpoint, py_obj, transformers))
  }
  switch(keras_class_type(py_obj),
         layer = make_r_fn.layer(endpoint, py_obj, transformers),
         make_r_fn.default(endpoint, py_obj, transformers))
}

endpoint_to_expr <- function(endpoint) {
  py_obj_expr <- endpoint |> str_split_1(fixed(".")) |>
    glue::backtick() |> str_flatten("$") |> str2lang()
  py_obj_expr
}

make_r_fn.default <- function(endpoint, py_obj, transformers) {
  # transformers <- get_arg_transformers(py_obj)
  # if(py_is(py_obj, keras$losses$BinaryCrossentropy)) browser()
  py_obj_expr <- endpoint_to_expr(endpoint)

  if (!length(transformers))
    transformers <- NULL

  as.function.default(c(formals(py_obj), bquote({
    args <- capture_args2(.(transformers))
    do.call(.(py_obj_expr), args)
  })))
}

make_r_fn.op <- function(endpoint, py_obj, transformers) {
  frmls <- formals(py_obj)
  py_obj_expr <- endpoint_to_expr(endpoint)
  syms <- lapply(names(frmls), as.symbol)
  cl <- as.call(c(py_obj_expr, syms))
  if(!length(transformers)) {
    fn <- as.function.default(c(frmls, cl))
  } else
    fn <- make_r_fn.default(endpoint, py_obj, transformers)

  fn

}

make_r_fn.activation <- function(endpoint,py_obj, transformers) {
  fn <- make_r_fn.default(endpoint, py_obj, transformers)
  attr(fn, "py_function_name") <- py_obj$`__name__`
  fn
}

make_r_fn.layer <- function(endpoint, py_obj, transformers) {

  ## build fn
  frmls <- formals(py_obj)
  frmls$self <- NULL
  py_obj_expr <- str2lang(str_replace_all(endpoint, fixed("."), "$"))


  ## first deal w/ special cases / exceptions

  if (grepl("layers.merging.", py_obj$`__module__`, fixed = TRUE)) {
    # merging layers Add, Subtract, Dot, etc:
    #   - accept unnamed args as tensors in ...
    #   - first arg is `inputs`, not `object`

    frmls <- c(alist(inputs = , ... =), frmls)
    frmls <- frmls[unique(names(frmls))] # remove dup `...` if present

    fn_body <- bquote({
      args <- capture_args2(.(transformers), ignore = c("...", "inputs"))
      dots <- split_dots_named_unnamed(list(...))
      if (missing(inputs))
        inputs <- NULL
      else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
      inputs <- c(inputs, dots$unnamed)
      args <- c(args, dots$named)

      layer <- do.call(.(py_obj_expr), args)

      if(length(inputs))
        layer(inputs)
      else
        layer
    })

  } else if (grepl(".rnn.", py_obj$`__module__`, fixed = TRUE) &&
             grepl("Cells?$", py_obj$`__name__`)) {
    # layer_gru_cell() and friends don't compose w/ `object`
    # TODO: consider renaming these in keras 3, maybe something like
    # rnn_cell_{gru,simple,stacked,lstm}()
    fn_body <- bquote({
      args <- capture_args2(.(transformers))
      do.call(.(py_obj_expr), args)
    })

  } else if (endpoint == "keras.layers.MultiHeadAttention") {
    # first arg is `inputs`, a list
    # TODO: in keras 3, change signature to:
    #    alist(query, key = query, value = key, ..., call_args = list())
    frmls <- c(alist(inputs = ), frmls)
    fn_body <- bquote({
      args <- capture_args2(.(transformers), ignore = "inputs")
      layer <- do.call(.(py_obj_expr), args)
      if (missing(inputs) || is.null(inputs))
        return(layer)
      if (!is.list(inputs))
        inputs <- list(inputs)
      do.call(layer, inputs)
    })

  } else {
    # default path for all other layers

    frmls <- c(alist(object = ), frmls)
    fn_body <- bquote({
      args <- capture_args2(.(transformers), ignore = "object")
      create_layer(.(py_obj_expr), object, args)
    })

  }

  fn <- as.function.default(c(frmls, fn_body), envir = emptyenv())
  fn <- rlang::zap_srcref(fn)

  fn
}


# ---- make, format, dump ----

mk_export <- function(endpoint) {

  # py parts
  py_obj <- py_eval(endpoint)
  name <- py_obj$`__name__`
  module <- py_obj$`__module__`
  # browser()
  docstring <- trim(py_obj$`__doc__` %||% "")
  type <- keras_class_type(py_obj)

  doc <- split_docstring_into_sections(docstring)

  # roxygen parts
  r_name <- make_r_name(endpoint, module)
  params <- parse_params_section(doc$arguments)
  tags <- make_roxygen_tags(endpoint, py_obj, type)

  if(!is.null(doc$call_arguments) &&
     endpoint != "keras.layers.Bidirectional") {
    doc$call_arguments %<>%
      parse_params_section() %>%
      {glue("- `{names(.)}`: {unname(.)}") }
      str_flatten_lines()
    ## This kidna works, but fails if there is a nested list within,
    ## like w/ layer_additive_attention
    # doc$call_arguments %<>%
    #   parse_params_section() %>%
    #   {glue("  \\item{<names(.)>}{<unname(.)>}",
    #         .open = "<", .close = ">")} %>%
    #   str_flatten_lines() %>%
    #   sprintf("\\describe{\n%s\n}", .)
    # browser()
  }

  # r wrapper parts
  arg_transformers <- get_arg_transformers(endpoint, py_obj, params)
  r_fn <- make_r_fn(endpoint, py_obj, type, arg_transformers)
#
  # fixes for special cases
  if (endpoint == "keras.layers.Lambda") {
    names(formals(r_fn)) %<>% replace_val("function", "f")
    names(params)        %<>% replace_val("function", "f")
  }

  # finish
  dump <- dump_keras_export(doc, params, tags, r_name, r_fn)

  as.list(environment())
}


dump_keras_export <- function(doc, params, tags, r_name, r_fn) {

  params <- params %>%
    { glue("@param {names(.)} {list_simplify(., ptype = '')}") } %>%
    str_flatten("\n")

  doc$arguments <- NULL

  # for(rd_sec in c("description", "details", "note")) {
  for(rd_sec in c("description")) {
    if(!is.null(sec <- doc[[rd_sec]]))
      sec <- str_flatten(c(glue("@{rd_sec}"), str_trim(sec)), "\n")
    doc[[rd_sec]] <- sec
  }

  main <- lapply(names(doc), \(nm) {
    md_heading <- if (nm %in% c("description", "title", "details", "note"))
      character() else
        snakecase::to_title_case(nm, prefix = "# ")
    str_flatten(c(md_heading, doc[[nm]]), "\n")
  })

  tags <- tags %>%
    {glue("@{names(.)} {list_simplify(.)}")} %>%
    str_flatten("\n")

  roxygen <- c(main, params, tags) %>%
    str_flatten("\n\n") %>% str_split_lines() %>%
    str_c("#' ", .) %>%
    str_flatten_lines()

  fn_def <- str_flatten_lines(str_c(r_name, " <-"),
                              deparse(r_fn))

  str_flatten_lines(roxygen, fn_def) |>
    str_split_lines() |> str_trim("right") |>
    str_flatten_lines()
}



mk_layer_activation_selu <- function() {
  selu <- mk_export("keras.activations.selu")
  selu$name <- "selu"
  selu$module <- keras$layers$ReLU$`__module__` %>%
    gsub("relu", "selu", ., ignore.case = TRUE)
  selu$endpoint <- "keras.layers.~selu" # pretend
  selu$r_name <- "layer_activation_selu"
  selu$r_fn <- zap_srcref(function(object, ...) {
    layer_activation(object, activation = "selu", ...)
  })
  selu$tags$inheritDotParams <- "layer_activation"
  selu$tags$family <- "activation layers"
  selu$params$object <- "tensor or model"

  selu$dump <- with(selu, dump_keras_export(doc, params,
                                            tags, r_name, r_fn))
  selu$type <- "layer"

  selu
}
