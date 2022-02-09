library(tibble)
library(purrr)
library(dplyr, warn.conflicts = FALSE)
library(reticulate)
library(envir)
library(tidyr, include.only = "unchop")
# attach_eval({
#   import_from(tidyr, unchop)
# })

# Sys.setenv(RETICULATE_PYTHON = "~/.local/share/r-miniconda/envs/tf-2.7-cpu/bin/python")
# Sys.setenv(RETICULATE_PYTHON = conda_python("tf-2.7-cpu"))


options(tibble.print_min = 200)


attach_eval({
  inspect <- reticulate::import("inspect")

  import_from(magrittr, `%<>%`)

  `%error%` <- function(x, y) tryCatch(x, error = function(e) y)

  `append1<-` <- function (x, value) {
    stopifnot(is.list(x) || identical(mode(x), mode(value)))
    x[[length(x) + 1L]] <- value
    x
  }

  find_keras_layer_dollar_call <- function(expr) {
    found <- list()
    find_it <- function(expr) {
      if (!is.call(expr))
        return(NULL)

      for (e in as.list(expr)) {
        if (is.call(e) && identical(e[[1]], quote(`$`)) &&
            grepl("^keras\\$layers\\$", s <- deparse1(e)))
          append1(found) <<- e
        find_it(e)
        }
      }

    find_it(expr)
    if(length(found) == 1) found[[1]] else found
  }

  v_py_id <- function(x) map_int(
    x, ~reticulate::py_id(.x) %||% stop() %error% NA_integer_
  )

  is_keras_layer <- py_run_string("
import tensorflow as tf
def is_keras_layer(x):
  try:
    return issubclass(x, tf.keras.layers.Layer)
  except:
    return False
")$is_keras_layer



  vdeparse1 <- function(x) {
    map_chr(x, ~ {if (is.null(.x)) "" else deparse1(.x)})
  }

})




py_df <-
  tibble(
    py_name = names(keras::keras$layers),
    py_obj = map(py_name, ~ keras::keras$layers[[.x]]),
    py_id = map_int(py_obj, reticulate::py_id)
  ) %>%
  group_by(py_id) %>%
  summarize(
    py_aliases = setdiff(py_name, py_obj[[1]]$`__name__`) %>%
      stringr::str_flatten(","),
    py_name = py_obj[[1]]$`__name__`,
    py_obj = py_obj[1],
  ) %>%
  ungroup() %>%
  # filter(!py_name %in% c("experimental")) %>%
  filter(!endsWith(py_name, "experimental")) %>%
  rowwise() %>%
  mutate(py_formals = list(keras:::py_formals(py_obj)),
         py_args = list(names(py_formals))) %>%
  ungroup()




r_df <- tibble(
  r_name = c(ls(pattern = "^layer_", envir = asNamespace("keras")),
             "bidirectional", "time_distributed"),
  r_fn = map(r_name, get, envir = asNamespace("keras")),
  r_formals = map(r_fn, ~ as.list(formals(.x))),
  r_args = map(r_formals, names),
  r_py_layer_expr = map(r_fn, ~ (find_keras_layer_dollar_call(body(.x))))
  ) %>%
  # filter(r_name == "layer_text_vectorization") %>%
  mutate_at("r_py_layer_expr", function(col)
            map(col, ~ { if (is.list(.x)) .x else list(.x) })) %>%
  unchop(r_py_layer_expr) %>%
  mutate(
    r_py_layer_name = map_chr(r_py_layer_expr, function(e) {
      if (!length(e)) return(NA_character_)
      sub("keras$layers$", "", deparse1(e), fixed = TRUE)
    }),
    r_py_obj = map(r_py_layer_expr, ~ try(eval(.x, asNamespace("keras")))),
    r_py_id = map_int(r_py_obj,
                      ~ reticulate::py_id(.x) %||% stop() %error% NA_integer_)
    )




# df <- full_join(r_df, py_df, c("r_py_layer_name" = "py_name"))
df <- full_join(r_df, py_df, c("r_py_id" = "py_id"))


missing_layers_df <- df %>%
  filter(is.na(r_name)) %>%
  filter(!py_name %in% c("serialize", "deserialize",
                         "Wrapper", "TimeDistributed", "Bidirectional",
                         "InputLayer", "Layer", "InputSpec"))


missing_layers_df %>%
  select(py_name)

# TF 2.8:
# # A tibble: 2 × 1
# py_name
# <chr>
#   1 DepthwiseConv1D
# 2 AbstractRNNCell

# Tensorflow version 2.7.0-rc0
# # A tibble: 1 × 1
#   py_name
#   <chr>
# 1 AbstractRNNCell

# # A tibble: 6 × 1
#   py_name
#   <chr>
# 1 RNN
# 2 AbstractRNNCell
# 3 SimpleRNNCell
# 4 GRUCell
# 5 StackedRNNCells
# 6 LSTMCell


if(FALSE) {

source("tools/make-wrapper2.R")
missing_layers_df$py_obj %>%
  lapply(new_layer_wrapper) %>%
  unlist() %>% str_flatten("\n\n\n") %>%
  {
    clipr::write_clip(.)
    cat(.)
  }
}


row2list <- function(x) lapply(x, \(col) if(is.list(col) && length(col) == 1) col[[1]] else col)

# r wrapper missing args
df %>%
  # filter(r_name == "layer_conv_1d") %>%
  # row2list()
  mutate(missing_in_r = map2(py_args, r_args, ~ setdiff(.x, c("self", .y)))) %>%
  relocate(missing_in_r) %>%
  mutate(across(c(missing_in_r, py_args, r_args), ~map_chr(.x, yasp::pcc))) %>%
  select(r_name, py_name, missing_in_r, r_args, py_args) %>%
  filter(missing_in_r != "...") %>%
  filter(missing_in_r != "") %>%
  identity()

# TODO: backend, applications
