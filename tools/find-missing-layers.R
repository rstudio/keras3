library(tibble)
library(purrr)
library(dplyr, warn.conflicts = FALSE)
library(reticulate)
library(envir)

options(tibble.print_min = 200)
# py_to_r.python.builtin.dict_items <- function(x) {
#   x <- py_eval("lambda x: [(k, v) for k, v in x]")(x)
#   out <- lapply(x, `[[`, 2L)
#   names(out) <- lapply(x, `[[`, 1L)
#   out
# }


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
  r_name = ls(pattern = "^layer_", envir = asNamespace("keras")),
  r_fn = map(r_name, get, envir = asNamespace("keras")),
  r_formals = map(r_fn, ~ as.list(formals)),
  r_args = map(r_formals, names),
  r_py_layer_expr = map(r_fn, ~ (find_keras_layer_dollar_call(body(.x))))
  ) %>%
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


# Error in py_get_attr_impl(x, name, silent) :
#   AttributeError: module 'keras.api._v2.keras.layers' has no attribute 'CuDNNGRU'
# Error in py_get_attr_impl(x, name, silent) :
#   AttributeError: module 'keras.api._v2.keras.layers' has no attribute 'CuDNNLSTM'



# df <- full_join(r_df, py_df, c("r_py_layer_name" = "py_name"))
df <- full_join(r_df, py_df, c("r_py_id" = "py_id"))


missing_layers_df <- df %>%
  filter(is.na(r_name)) %>%
  filter(!py_name %in% c("serialize", "deserialize",
                         "Wrapper", "TimeDistributed", "Bidirectional",
                         "InputLayer", "Layer", "InputSpec"))

source("tools/make-wrapper2.R")

missing_layers_df %>%
  select(py_name)
# # A tibble: 37 × 1
#    py_name
#    <chr>
#    image preprocessing
# 12 Resizing
# 13 CenterCrop
# 15 Rescaling
#
#    # image augmentation
# 14 RandomCrop
# 16 RandomFlip
# 17 RandomTranslation
# 18 RandomRotation
# 19 RandomZoom
# 21 RandomHeight
# 22 RandomWidth
# 20 RandomContrast
#
# # categorical features preprocessing
# 23 CategoryEncoding
# 25 Hashing
# 28 StringLookup
# 26 IntegerLookup
#
# # numerical features preprocessing
# 24 Discretization
# 27 Normalization
#
# # Text preprocessing
# 29 TextVectorization
#
#
# 30 StackedRNNCells
# 31 RNN
# 32 AbstractRNNCell
# 33 SimpleRNNCell
# 34 GRUCell
# 35 LSTMCell
# 36 ConvLSTM1D
# 37 ConvLSTM3D



missing_layers_df$py_obj %>%
  .[1:18] %>%
  lapply(new_layer_wrapper) %>%
  unlist() %>% str_flatten("\n\n\n") %>%
  {
    clipr::write_clip(.)
    cat(.)
  }
