library(tibble)
library(purrr)
library(dplyr, warn.conflicts = FALSE)
library(reticulate)
library(envir)

# keras::install_keras(envname = "tf-2.6-cpu")
# tools/setup-test-envs.R
# use_miniconda("tf-2.6-cpu", required=TRUE)

options(tibble.print_min = 100)
py_to_r_python.builtin.dict_items <- function(x) {
  x <- py_eval("lambda x: [(k, v) for k, v in x]")(x)
  out <- lapply(x, `[[`, 2L)
  names(out) <- lapply(x, `[[`, 1L)
  out
}


attach_eval({
  inspect <- reticulate::import("inspect")

  import_from(magrittr, `%<>%`)

  `%error%` <- function(x, y) tryCatch(x, error = function(e) y)

  find_first_arg_of_call <- function(expr, call = quote(create_layer)) {
    find_it <- function(expr) {
      if(!is.call(expr)) return(NULL)
      for(e in as.list(expr))
        if(is.call(e) && identical(e[[1]], call))
          return(e[[2]])
      else {
        if(!is.null(it <- find_it(e)))
          return(it)
      }
    }
    find_it(expr)
  }

})


default_ignore <- c("self", "kwargs")
DF <-
  # ls(pattern = "^layer_", envir = asNamespace("keras")) %>%
  # setdiff(c("layer_cudnn_gru", "layer_cudnn_lstm")) %>%
  # filter(r_name == "layer_text_vectorization") %>%
  "layer_text_vectorization" %>%
  set_names() %>%
  lapply(function(r_func_nm) {
    r_func <- get(r_func_nm, envir = asNamespace("keras"))
    r_args <- names(formals(r_func))
    py_layer <- eval(find_first_arg_of_call(body(r_func)),
                     asNamespace("keras")) %error% NULL

    py_init_formals <-
      # keras:::py_formals(py_layer) %error% NULL
      # inspect$signature(py_layer$`__init__`)$parameters$items() %error% NULL
      py_to_r_python.builtin.dict_items(inspect$signature(py_layer$`__init__`)$parameters$items()) %error% NULL
    py_init_args <- names(py_init_formals) %error% NULL
    py_init_defaults <- py_init_formals |> lapply(function(x) x$default) %error% NULL
    lst(r_func_nm, r_func, r_args, py_layer, py_init_args, py_init_defaults)
  })  %>%
  transpose() %>%
  as_tibble() %>%
  mutate_at("r_func_nm", unlist) %>%
  mutate(missing_in_r_func_args = map2_chr(r_args, py_init_args,
                                           ~ setdiff(.y, c(.x, default_ignore)) %>%
                                             paste(collapse = ", "))) %>%
  filter(r_func_nm != "layer_activation_selu") %>%   # Not a real layer on keras side
  # filter(!r_func_nm %in% c("layer_cudnn_gru", "layer_cudnn_lstm"))
  identity()



DF$missing_in_r_func_args[DF$r_func_nm == "layer_lambda"] %<>% sub("function", "", .)

DF %>%
  filter(missing_in_r_func_args != "") %>%
  select(r_func_nm, missing_in_r_func_args) %>%
  print(n = Inf)

# tf 2.7
# A tibble: 0 × 2
# … with 2 variables: r_func_nm <chr>, missing_in_r_func_args <chr>

# tf 2.6
# # A tibble: 5 × 2
#   r_func_nm                       missing_in_r_func_args
#   <chr>                           <chr>
# 1 layer_global_average_pooling_2d keepdims
# 2 layer_global_average_pooling_3d keepdims
# 3 layer_global_max_pooling_1d     keepdims
# 4 layer_global_max_pooling_2d     keepdims
# 5 layer_global_max_pooling_3d     keepdims

# tf 2.5
# # A tibble: 12 x 2
#    r_func_nm                  missing_in_r_func_args
#    <chr>                      <chr>
#  1 layer_conv_1d              groups
#  2 layer_conv_2d              groups
#  3 layer_conv_3d              groups
#  4 layer_conv_3d_transpose    dilation_rate
#  5 layer_conv_lstm_2d         return_state
#  6 layer_depthwise_conv_2d    dilation_rate
#  7 layer_gru                  time_major
#  8 layer_locally_connected_1d implementation
#  9 layer_locally_connected_2d implementation
# 10 layer_lstm                 time_major
# 11 layer_max_pooling_1d       data_format
# 12 layer_text_vectorization   vocabulary

#> # A tibble: 19 x 2
#>    r_func_nm            missing_in_r_func_args
#>    <chr>                <chr>
#>  1 layer_activation_se… activation
#>  2 layer_batch_normali… renorm, renorm_clipping, renorm_momentum, fused, v…
#>  3 layer_conv_2d_trans… stride, out_pad
#>  4 layer_conv_3d_trans… stride, out_pad
#>  5 layer_conv_lstm_2d   cell
#>  6 layer_cropping_2d    height_cropping, width_cropping
#>  7 layer_cropping_3d    dim1_cropping, dim2_cropping, dim3_cropping
#>  8 layer_cudnn_gru      go_backwards, cell_spec
#>  9 layer_cudnn_lstm     go_backwards, cell_spec
#> 10 layer_embedding      dtype
#> 11 layer_gru            implementation, cell
#> 12 layer_lambda         function, function_args
#> 13 layer_locally_conne… implementation
#> 14 layer_locally_conne… implementation
#> 15 layer_lstm           implementation, cell
#> 16 layer_max_pooling_1d data_format
#> 17 layer_simple_rnn     cell
#> 18 layer_zero_padding_… height_padding, width_padding
#> 19 layer_zero_padding_… dim1_padding, dim2_padding, dim3_padding
