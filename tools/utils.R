# ---- setup attach  ----

library_stable <-
  function(package,
           lifecycle_exclude = c(
             "superseded",
             "deprecated",
             "questioning",
             "defunct",
             "experimental",
             "soft-deprecated",
             "retired"
           ), ...,
           exclude = NULL) {

    package <- deparse1(substitute(package))
    exclude <- lifecycle::pkg_lifecycle_statuses(
      package, lifecycle_exclude)$fun |> c(exclude)

    # message(package, ", excludeing: ", paste0(exclude, collapse = ", "))
    library(package, character.only = TRUE, ...,
            exclude = exclude)
  }

library_stable(readr)
library_stable(stringr)
library_stable(tibble)
library_stable(tidyr)
library_stable(dplyr, warn.conflicts = FALSE)
library_stable(rlang)
library_stable(purrr)
library_stable(stringr)
library_stable(glue)

library(envir)
library(commafree)
library(magrittr, include.only = c("%>%", "%<>%"))
library(reticulate)
library(assertthat, include.only = c("assert_that"))

attach_eval({
  import_from(TKutils, `%error%`)

  is_scalar <- function(x) identical(length(x), 1L)

  replace_val <- function (x, old, new) {
    if (!is_scalar(new))
      stop("Unexpected length of replacement value in replace_val().\n",
           "`new` must be length 1, not ", length(new))
    x[x %in% old] <- new
    x
  }

  py_is <- function(x, y) identical(py_id(x), py_id(y))

  str_split1_on_first <- function(x, pattern, ...) {
    stopifnot(length(x) == 1, is.character(x))
    regmatches(x, regexpr(pattern, x, ...), invert = TRUE)[[1L]]
  }

  str_flatten_lines <- function(..., na.rm = FALSE) {
    stringr::str_flatten(c(...), na.rm = na.rm, collapse = "\n")
  }

  str_split_lines <- function(...) {
    x <- c(...)
    if(length(x) > 1)
      x <- str_flatten_lines(x)
    stringr::str_split_1(x, "\n")
  }

  `append<-` <- function(x, after = length(x), value) {
    append(x, value, after)
  }

  unwhich <- function(x, len) {
    lgl <- logical(len)
    lgl[x] <- TRUE
    lgl
  }

  in_recursive_call <- function() {
    fn <- sys.call(-1)[[1]]
    for(cl in head(sys.calls(), -2))
      if(identical(cl[[1]], fn))
        return(TRUE)
    return(FALSE)
  }

  # use like msg(user = ...), assistant =
  chat_messages <- function(...) {
    x <- rlang::dots_list(..., .named = TRUE, .ignore_empty = "all")
    stopifnot(all(names(x) %in% c("system", "user", "assistant")))
    unname(imap(x, \(content, role)
                list(role = role,
                     content = trim(str_flatten_lines(content)))))
  }

  str_detect_non_ascii <- function(chr) {
    map_lgl(chr, \(x) {
      if (Encoding(x) == "unknown") {
        Encoding(x) <- "UTF-8"
      }
      Encoding(x) != "unknown"
    })
  }

  str_remove_non_ascii <- function(chr) {
    i <- str_detect_non_ascii(chr)
    if (any(i)) {
      chr[i] <- chr[i] |> map_chr(\(x) {
        x |>
          iconv(to = "ASCII") |>
          str_replace_all(fixed("??"), "")
      })
    }
    chr
  }

  # ignore empty trailing arg
  c <- function(...)
    base::do.call(base::c, rlang::list2(...), quote = TRUE)

}, mask.ok = "c")

# ---- venv ----
# reticulate::virtualenv_remove("r-keras")
# if(!virtualenv_exists("r-keras")) install_keras()

use_virtualenv("r-keras", required = TRUE)

inspect <- import("inspect")

# keras <- import("keras_core")
# keras <- import("tensorflow.keras")
keras <- import("keras")
local({
  `__main__` <- reticulate::import_main()
  `__main__`$keras <- keras
})

source_python("tools/common.py") # keras_class_type()
rm(r) # TODO: fix in reticulate, don't export r


# ---- find ----


list_endpoints <- function(
    module = "keras", max_depth = 4,
    skip = "keras.src",
    skip_regex = sprintf("\\.%s$", c("experimental", "deserialize", "serialize", "get"))) {

  .list_endpoints <- function(.module, depth) {
    if (depth > max_depth) return()

    module_py_obj <- py_eval(.module)
    lapply(names(module_py_obj), \(nm) {
      endpoint <- paste0(.module, ".", nm)
      if(length(endpoint) > 1) browser()
      if (endpoint %in% skip) return()
      if (any(str_detect(endpoint, skip_regex))) return()
      # message(endpoint)
      endpoint_py_obj <- module_py_obj[[nm]]
      if (inherits(endpoint_py_obj, "python.builtin.module"))
        return(.list_endpoints(endpoint, depth = depth + 1L))
      if (inherits(endpoint_py_obj, c("python.builtin.type",
                                      "python.builtin.function")))
        return(endpoint)
      NULL
    })
  }

  unlist(lapply(module, .list_endpoints, depth = 1L))
}

if(FALSE) {
  list_endpoints(c("keras.activations", "keras.regularizers"))

  list_endpoints(c("keras.activations", "keras.regularizers"))

  all_endpoints <- list_endpoints()

  all_endpoints %>%
    grep("ImageDataGenerator", ., value = T)

  all_endpoints %>%
    grep("Image", ., value = T, ignore.case = TRUE)

}



# ---- docstrings ----

known_section_headings <- c %(% {
  "Args:"
  "Arguments:"
  "Attributes:"
  "Returns:"
  "Raises:"
  "Input shape:"
  "Inputs:"
  # "Input:",
  "Output shape:"
  # "Outputs:",
  "Output:"
  "Call arguments:"
  "Call Args:"
  "Returns:"
  "Example:"
  "References:"
  "Reference:"
  "Examples:"
  "Notes:"
  "Note:"
  "Usage:"
  # "Standalone usage:"
}


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
  docstring <- docstring |> backtick_not_links()

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

  # browser()

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
  maybe_new_sec %<>% .[. > which.max(m == "arguments")]
  if(length(maybe_new_sec)) {
    m[maybe_new_sec[1]:length(m)] %<>% replace_val("reference", "description")
    append(m, after = maybe_new_sec[1]) <- "description"

    x[maybe_new_sec[1]] %<>% paste0("\n", .)
    x %<>% str_split_lines()
  }

  sections <- split(x, m) |>
    imap(\(s, nm) {
      if (!nm %in% c("description", "title", "details")) # "note"
        s <- s[-1] # drop heading
      s |> str_trim("right") |> str_flatten("\n") |> trim()
    })

  sections <- sections[unique(c(m))] # split() reorders

  sections
}


vararg_paramater_names <- function(py_obj) {
  params <- inspect$signature(py_obj)$parameters$values()
  params %>%
    as_iterator() %>% iterate(\(p) {
      if (p$kind %in% c(inspect$Parameter$VAR_POSITIONAL,
                        inspect$Parameter$VAR_KEYWORD))
        p$name
      else
        NULL
    }) %>% unlist()
}


parse_params_section <- function(docstring_args_section, treat_as_dots = NULL) {

  for (chk_empty_fn in c(is.null, is.na, partial(`==`, "")))
    if(chk_empty_fn(docstring_args_section))
      return(list())

  if(!is_scalar(docstring_args_section))
    docstring_args_section %<>% str_flatten("\n")


  # if(str_detect(docstring_args_section, fixed("recall: A scalar value in range `[0,")))
  #   browser()

  x <- docstring_args_section |>
    c("") |> # append final new line to ensure glue::trim() still works when length(x) == 1
    str_flatten("\n") |> glue::trim() |> str_split_1("\n")

  m <- str_match(x, "^\\s{0,3}(?<name>[*]{0,2}[A-Za-z0-9_]+) ?: *(?<desc>.*)$")

  description <- m[,"desc"] %|% x
  param_name <- m[,"name"]

  not_param <- which(param_name == "Default")
  if(length(not_param)) {
    description[not_param] <- x[not_param]
    param_name[not_param] <- NA
  }

  param_name[startsWith(param_name, "*")] <- "..."
  # param_name[param_name == "kwargs"] <- "..."
  param_name[param_name %in% treat_as_dots] <- "..."

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
  i <- 0L
  while(i < length(docstring)) {
    i <- i + 1L

    if(startsWith(docstring0[[i]], "```")) {
      in_code_block <- TRUE
      while(i < length(docstring0)) {
        if (startsWith(docstring0[[i <- i + 1L]], "```")) {
          in_code_block <- FALSE
          break
        }
      }
      next
    }

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
        # } else if(docstring0[[i]] == "```") {
        #   # code block closed in docstring
        #   in_code_block <- FALSE
      } else
        # tidy up example output
        docstring[[i]] <- str_c("# ", docstring[[i]])
    }
  }
  if(in_code_block)
    docstring <- c(docstring, "```\n")

  docstring <- str_flatten(docstring, collapse = "\n")

}


str_split_1_inclusive <- function(string, pattern) {
  unlist(stringi::stri_split_regex(
    string, pattern =
      paste0("(?<=(", pattern, "))",  # match before pattern
             "|",                     # or
             "(?=(", pattern, "))")))  # match after pattern
}


backtick_not_links <- function(d) {

  d %>%
    str_split_1_inclusive("\n```") %>%
    as.list() %>%
    {
      # mark prose and code sections.
      type <- "prose"
      for(i in seq_along(.)) {
        if(.[[i]] == "\n```") {
          names(.)[i] <- "delim"
          type <- switch(type, prose = "code", code = "prose")
          next
        }

        if(type == "prose") {
          s <- str_split_1_inclusive(.[[i]], "`")
          stype <- "prose"
          for(si in seq_along(s)) {
            if(s[[si]] == "`") {
              names(s)[si] <- "delim"
              stype <- switch(stype, prose = "code", code = "prose")
              next
            }
            names(s)[si] <- stype
          }
          .[i] <- list(s)
          next
        }
        if(type == "code")
          names(.)[i] <- "code"
      }
      . <- unlist(.)
      names(.) <- gsub("NA.", "", names(.), fixed = TRUE)
      .
    } %>% {
      # browser()
      i <- which(names(.) == "prose")
      names(.) <- NULL
      .[i] <- .backtick_not_links(.[i])
      .
    } %>%
    str_flatten()

}


.backtick_not_links <- function(x) {
  # if(any(str_detect(x, fixed("["))))
  # browser()
  #     [0-9a-zA-Z,\[\]\ -><.]+  # Capture numbers, brackets, and specific symbols
  re <- regex(comments = TRUE, pattern = r"--(
      (                 # Open capture group for main pattern
      [^\ \n]*      # anything not a space or newline
      \[              # Match an opening square bracket
          .+            # capture anything greedily
        \]              # Match a closing square bracket
      [^\ \n\(]*       #  # anything not a space or bounary or open parens
      )                 # Close capture group
      (?!\()            # Lookahead to ensure no opening parenthesis follows
    )--")
  rep <- r"--(`\1`)--"
  str_replace_all(x, re, rep)
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

  type <- keras_class_type(py_eval(endpoint))

  # TODO: this func is due a refactor
  #   - we should only call snake_case() on camelcase objects (e.g.,
  #     no need to call it on ops.
  #   - handle prefixes better
  # manual renames
  if(!is.null(r_name <- switch %(% { endpoint
    # "keras.preprocessing.image.array_to_img" = "image_from_array"
    # "keras.preprocessing.image.img_to_array" = "image_to_array"
    # "keras.preprocessing.image.load_img" =  "image_load"
    # "keras.preprocessing.image.save_img" = "image_array_save"
    # "keras.preprocessing.sequence.pad_sequences" = "pad_sequences"

    "keras.utils.array_to_img" = "image_from_array"
    "keras.utils.img_to_array" = "image_to_array"
    "keras.utils.load_img" =  "image_load"
    "keras.utils.save_img" = "image_array_save"
    "keras.utils.pad_sequences" = "pad_sequences"

    "keras.layers.LayerNormalization" = "layer_layer_normalization"

    "keras.utils.FeatureSpace" = "layer_feature_space"
    # TODO: why is FeatureSpace not exported to keras.layers.FeatureSpace?
    # Does instantiation and composition in one call make sense, or
    # does the need for adapt() throw a wrench in the works (and mean that
    # using compose_layer() doesn't make sense)...
    # maybe this should have a name like "preprocess_feature_space()" or
    # layer_preprocess_feature_space()? or


    "keras.ops.in_top_k" = "k_in_top_k"
    "keras.ops.top_k" = "k_top_k"

    "keras.random.randint" = "random_integer"
    # "keras.losses.LogCosh" = "loss_logcosh"
    NULL
  })) return(r_name)

  x <- endpoint |> str_split_1(fixed("."))
  x <- lapply(x, function(.x) switch(.x,
                                "keras" = character(),
                                "preprocessing" = character(),
                                "utils" = character(),
                                # "preprocessing" = character(),
                                ops = "k",
                                .x
                                ))

  name <- x[length(x)]
  prefix <- x[-length(x)] |> unlist() |>
    str_replace("e?s$", "") |> str_flatten("_")

  if(type == "learning_rate_schedule")
    prefix <- "learning_rate_schedule"

  prefix <- switch(prefix,
                   # "random" = "k_random",
                   # "config" = "k_config",

                   "ops" = "k",
                   prefix)

  # if(endpoint == "keras.preprocessing.image_dataset_from_directory")
  #   browser()
  # if(endpoint |> startsWith("keras.ops."))
  #   endpoint %<>% str_replace(fixed("keras.ops."), "keras.k.")
  # if(endpoint |> startsWith("keras.preprocessing."))
  #   endpoint %<>% str_replace(fixed(".preprocessing."), ".")


  # if(endpoint == ) return("")

  # x <- endpoint |>
  #   reticulate:::str_drop_prefix("keras.") |>
  #   str_split_1(fixed("."))

  # name <- x[length(x)] # __name__
  # x <- x[-length(x)] # submodules
  # x <- x[nzchar(x)]

  # "keras.optimizers.schedules.CosineDecay"
  # "optimizer_schedule_cosine_decay"
  # "learning_rate_schedule_cosine_decay"

  # if(length(x) >= 2)
  #   x <- x[-1] # drop "keras" from "keras.layers.Dense
  # if(!length(x))
  #   prefix <- x()

  # if(type == "learning_rate_schedule")
  #   prefix <- "learning_rate_schedule"
  # else {
  #   # x <-
  #   # if(length(x) && !is_scalar(x))
  #
  #     # browser()
  #   prefix <- x |> str_replace("e?s$", "") |> str_flatten("_")
  # }

  name <- name |>
    str_replace("NaN", "Nan") |>
    str_replace("RMSprop$", "Rmsprop") |>
    str_replace("ReLU$", "Relu") |>
    str_replace("ResNet", "Resnet") |>
    str_replace("ConvNeXt", "Convnext") |>
    str_replace("XLarge", "Xlarge") |>

    str_replace("IoU", "Iou") |>
    str_replace("FBeta", "Fbeta")
    # str_replace("FBeta", "Fbeta") |>
    # str_replace("EfficientNet", "Efficientnet") |>

  if(str_detect(name, "[[:upper:]]"))
    name %<>% snakecase::to_snake_case()

  name <- name |>

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

    str_replace("f_1", "f1") |>
    str_replace("r_2", "r2") |>
    # str_replace("log_2", "log2") |>
    # str_replace("log_10", "log10") |>
    # str_replace("log_1_p", "log1p") |>
    # str_replace("relu_6", "relu6") |>

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


  # if (prefix != "k") {
  #   name %<>%
  #     str_replace(glue("_?{prefix}_?"), "_") %>%
  #     str_replace("^_", "") %>%
  #     str_replace_all("_+", "_") %>%
  #     str_replace("_$", "")
  # }
  # if (length(prefix) && prefix != "")
    name <- str_flatten(c(prefix, name), collapse = "_")

  name %<>%
    str_split_1("_") %>%
    unique() %>%
    .[nzchar(.)] %>%
    str_flatten("_")

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
    # if(endpoint == "keras.random.randint") browser()
    lapply(args, function(fn) {
      # if(grepl("normalize_padding", fn)) browser()
      if(is.null(fn)) return(fn)
      fn <- str2lang(fn)
      if (is.call(fn) && !identical(fn[[1]], quote(`function`)))
        fn <- as.function.default(c(alist(x =), fn))
      fn
    })
  })


get_arg_transformers <- function(endpoint, py_obj = py_eval(endpoint), params = NULL) {

  if(is.null(params)) {
    params <-  get_fixed_docstring(endpoint) |> trim() |>
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

  # if(endpoint == "keras.random.randint") {
  #   frmls <- formals(keras$random$randint)
  #   frmls$minval <- 0L
  #   frmls$maxval <- 1L
  #   return(as.function.default(, quote({
  #     args <- capture_args2(list(shape = normalize_shape, seed = as_integer))
  #     do.call(keras$random$randint, args)
  #   }))
  # }


  switch(keras_class_type(py_obj),
         layer = make_r_fn.layer(endpoint, py_obj, transformers),
         metric = ,
         loss = make_r_fn.loss(endpoint, py_obj, transformers),
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

  frmls <- formals(py_obj)
  body <- bquote({
    args <- capture_args2(.(transformers))
    do.call(.(py_obj_expr), args)
  })

  # if(endpoint == "keras.preprocessing.image.save_img")
  if(endpoint == "keras.utils.save_img")
    frmls <- frmls[unique(c("x", "path", names(frmls)))] # swap so img is first arg, better for pipe
  # frmls <- frmls[c(2, 1, 3:length(frmls))] # swap so img is first arg, better for pipe

  as.function.default(c(frmls, body))
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


make_r_fn.loss <- function(endpoint, py_obj, transformers) {

  # see if there is a function counterpart handle to the class handle

  name <- py_obj$`__name__`
  endpoint_fn <- str_replace(endpoint, name, snakecase::to_snake_case(name))

  py_obj_fn <- tryCatch(py_eval(endpoint_fn),
                        python.builtin.AttributeError = function(e) NULL)

  if(is.null(py_obj_fn)) {
    fn <- make_r_fn.default(endpoint, py_obj, transformers)
    formals(fn) <- formals(py_obj) %>%
      c(alist(... =), .) %>% .[unique(names(.))]
    return(fn)
  }

  py_obj_expr <- endpoint_to_expr(endpoint)
  frmls <- formals(py_obj)
  frmls <- c(alist(... = ), formals(py_obj))
  frmls <- frmls[unique(names(frmls))]

  #%error% browser()
  ### browser() if we need a simple fallback in case there is a loss with only a
  ### class handle and not a matching function handle
  py_obj_expr_fn <- endpoint_to_expr(endpoint_fn)
  transformers <- modifyList(transformers %||% list(),
                             get_arg_transformers(endpoint_fn) %||% list()) %error% browser()
  if(!length(transformers))
    transformers <- NULL
  frmls <- c(formals(py_obj_fn), frmls)
  frmls <- frmls[unique(names(frmls))]
  stopifnot(names(frmls)[1:2] == c("y_true", "y_pred"))

  body <- bquote({
    args <- capture_args2(.(transformers))
    callable <- if (missing(y_true) && missing(y_pred))
      .(py_obj_expr) else .(py_obj_expr_fn)
    do.call(callable, args)
  })


  fn <- as.function.default(c(frmls, body))
  attr(fn, "py_function_name") <- py_obj_fn$`__name__`

  # TODO: compile should check for the bare metric/loss R wrapper being passed,
  # and maybe unwrap it to avoid the R overhead.

  # TODO: dump r docstring for the r function wrapper too
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

  } else if (endpoint == "keras.layers.Lambda") {
    # rename `function` -> `func`. because roxygen gets very confused
    # if you have an R parser reserved word as an arg name
    names(frmls) %<>% replace_val("function", "f")
    frmls <- c(alist(object = ), frmls)
    fn_body <- bquote({
      args <- capture_args2(.(transformers), ignore = "object")
      names(args)[match("f", names(args))] <- "function"
      create_layer(.(py_obj_expr), object, args)
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


param_augment_registry <- yaml::read_yaml("tools/param-descripations.yml") %>%
  map_depth(2, ~.x |> str_flatten_lines() |> str_squish())


# ---- make, format, dump ----

known_metrics_without_function_handles <- c %(% {
  "keras.metrics.AUC"
  "keras.metrics.BinaryIoU"
  "keras.metrics.CosineSimilarity"
  "keras.metrics.F1Score"
  "keras.metrics.FalseNegatives"
  "keras.metrics.FalsePositives"
  "keras.metrics.FBetaScore"
  "keras.metrics.IoU"
  "keras.metrics.LogCoshError"
  "keras.metrics.Mean"
  "keras.metrics.MeanIoU"
  "keras.metrics.MeanMetricWrapper"
  "keras.metrics.OneHotIoU"
  "keras.metrics.OneHotMeanIoU"
  "keras.metrics.Precision"
  "keras.metrics.PrecisionAtRecall"
  "keras.metrics.R2Score"
  "keras.metrics.Recall"
  "keras.metrics.RecallAtPrecision"
  "keras.metrics.RootMeanSquaredError"
  "keras.metrics.SensitivityAtSpecificity"
  "keras.metrics.SpecificityAtSensitivity"
  "keras.metrics.Sum"
  "keras.metrics.TrueNegatives"
  "keras.metrics.TruePositives"
}


mk_export <- function(endpoint) {

  # py parts
  py_obj <- py_eval(endpoint)
  name <- py_obj$`__name__`
  module <- py_obj$`__module__`
  docstring <- get_fixed_docstring(endpoint)
  type <- keras_class_type(py_obj)

  doc <- split_docstring_into_sections(docstring)



  # roxygen parts
  r_name <- make_r_name(endpoint, module)
  params <- parse_params_section(doc$arguments, treat_as_dots = vararg_paramater_names(py_obj))

  if(endpoint == "keras.layers.Lambda") {
    names(params)[match("function", names(params))] <- "f"
  }

  if ((endpoint |> startsWith("keras.losses.") ||
       endpoint |> startsWith("keras.metrics.")) &&
      inherits(py_obj, "python.builtin.type"))
    local({
      endpoint2 <- str_replace(endpoint, name, snakecase::to_snake_case(name))
      py_obj2 <- NULL
      tryCatch(
        py_obj2 <- py_eval(endpoint2),
        python.builtin.AttributeError = function(e) {

          if(!endpoint %in% known_metrics_without_function_handles)
            message(endpoint2, " not found")
          # message('"', endpoint, '"')
        }
      )
      if(is.null(py_obj2)) return()

      ep2 <- mk_export(endpoint2)
      params <<- modifyList(params, ep2$params)
      for(section in setdiff(names(ep2$doc), "title"))
        doc[[section]] %<>% str_flatten_lines(ep2$doc[[section]], .)
      doc <<- doc
    })

  tags <- make_roxygen_tags(endpoint, py_obj, type)

  if(!is.null(doc$call_arguments) &&
     endpoint != "keras.layers.Bidirectional") {
    # convert `Call Arguments:` section to a md list.
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

  local({
    if (length(undocumented_params <-
               setdiff(names(formals(r_fn)),
                       unlist(strsplit(names(params) %||% character(), ","))))) {

      # if(endpoint == "keras.layers.Add") browser()
      maybe_add_params <- param_augment_registry %>%
        .[str_detect(endpoint, glob2rx(names(.))) |
            str_detect(paste0(module, ".", name), glob2rx(names(.)))] %>%
        unname() %>% unlist(recursive = FALSE)

      # TODO:, handle 'inputs,...'
      for(new_param_name in intersect(names(maybe_add_params), undocumented_params))
        params[[new_param_name]] <<- maybe_add_params[[new_param_name]]

    }
  })

  if(!in_recursive_call())
    local({
      if (length(undocumented_params <-
                 setdiff(names(formals(r_fn)),
                         unlist(strsplit(names(params) %||% character(), ","))))) {
        # browser()
        x <- list2("{endpoint}" := map(set_names(undocumented_params), ~"see description"))
        writeLines(yaml::as.yaml(x))
        # message(endpoint, ":")
        # writeLines(paste("  ", undocumented_params, ":", "see description"))
      }
    })

  local({
    frmls <- formals(r_fn)
    params_documented_but_missing <- names(params) |> setdiff(names(frmls))
    if(length(params_documented_but_missing)) {
      frmls <- c(frmls, lapply(purrr::set_names(params_documented_but_missing), \(n) NULL))
      formals(r_fn) <<- frmls
    }
  })



  #
  # fixes for special cases
  # if (endpoint == "keras.layers.Lambda") {
  #   names(formals(r_fn)) %<>% replace_val("function", "f")
  #   names(params)        %<>% replace_val("function", "f")
  # }

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

  # [^(]             # is not followed by an opening parens (indicating a md link)
  # r"---((?<!`)\[([0-9,\-\[\]]+)\](?!\())---"
  # m <- "(?<!`)\\[([0-9,\\- >.\\[\\]]+)\\](?!\\()"
  # r <- "`\\[\\1\\]`"
  # doc$description %<>% gsub(m, r, ., perl = TRUE)



  main <- lapply(names(doc), \(nm) {
    md_heading <- if (nm %in% c("description", "title", "details")) # "note"
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
    str_flatten_lines() %>%
    str_split_lines() %>%
    str_trim("right") %>%
    str_flatten_lines()

  # compact roxygen
  roxygen %<>% {
    while (nchar(.) != nchar(. <- gsub(strrep("#'\n", 2), "#'\n", ., fixed = TRUE))) {} # TODO: do this in dump
    while (nchar(.) != nchar(. <- gsub(strrep("\n", 3), "\n\n", ., fixed = TRUE))) {}
    .
  }

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



get_docstring <-
get_fixed_docstring <- function(endpoint) {

  d <- inspect$getdoc(py_eval(endpoint)) %||% ""

  if(d == "") {
    return(switch(endpoint,
      "keras.random.dropout" = r"{random dropout

        Randomly set some portion of values in the tensor to 0.}",
      ""))
  }

  d %<>% str_remove_non_ascii()
  # if(str_detect_non_ascii(d))
  #   d <- d %>% iconv(to = "ASCII") %>% gsub("??", "", ., fixed = TRUE)

  replace <- function(old_txt, new_txt) {
    d <<- stringi::stri_replace_first_fixed(d, old_txt, new_txt)
  }

  switch %(% { endpoint

    keras.ops.average_pool = `,`
    keras.ops.depthwise_conv = `,`
    keras.ops.separable_conv = `,`
    keras.ops.conv_transpose = replace(
      "`(batch_size,)` + inputs_spatial_shape ",
      "`(batch_size,) + inputs_spatial_shape ")

    keras.ops.scatter = replace(
      "    updates: A tensor, the values to be set at `indices`.",
      "    values: A tensor, the values to be set at `indices`."
    )

    keras.metrics.RecallAtPrecision = replace(
      "        num_thresholds: (Optional) Defaults to 200. The number of thresholds",
      "    num_thresholds: (Optional) Defaults to 200. The number of thresholds"
    )

    # keras.ops.einsum = replace(
    #   "operands: The operands to compute the Einstein sum of.",
    #   "*args: The operands to compute the Einstein sum of.")


    NULL
  }


  d <- trim(d)
  d %<>% str_split_lines()
  if("Standalone usage:" %in% d &&
     !any(c("Usage:", "Examples:") %in% d)) {
    d %<>% replace_val("Standalone usage:", "Usage:\n\nStandalone usage:")
  }
  d %<>% str_flatten_lines()
  # d %>% str_split_lines() %>%
  # d %>% str_split_lines() %>%
  # d <- str_replace("^(\\s*)Standalone usage:$", "\\1Examples:\n\\1Standalone usage:")
  # replace("Standalone usage:\n", "Examples:\n\nStandalone usage:\n")
  d
}

# ---- misc utils ----



# rename2(list(a = "b", a = z))


# rx <- regex(comments = TRUE, pattern = r"--(
#       `?                # Optional backtick at the start
#       (                 # Open capture group for main pattern
#         \[              # Match an opening square bracket
#         [0-9,\[\]\ -><.]+  # Capture numbers, brackets, and specific symbols
#         \]              # Match a closing square bracket
#       )                 # Close capture group
#       `?                # Optional backtick at the end
#       (?!\()            # Lookahead to ensure no opening parenthesis follows
#     )--")
# rep <- r"--(`\1`)--"
# doc$description %<>% str_replace_all(rx, rep)

#
# x <- "e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`"
# rx <- regex(comments = TRUE, r"--(
#       `?(            # open capture group, without backtick
#       \[             # opening bracket, maybe starts with a backtick
#       [0-9,\[\]\ -><.]+  # anything that looks like a number range
#       \]
#       )             # close capture group
#       `?             # closing bracket maybe w backtick
#       (?!\()           # is not followed by an opening parens indicating a md link
#   )--")
# rep <- "`\\1`"
# str_replace_all(x, rx, rep)
