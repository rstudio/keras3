#!source envir::attach_source(.file)
if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

readRenviron("~/.Renviron")
import("os")$environ$update(list(OPENAI_API_KEY = Sys.getenv("OPENAI_API_KEY")))
openai <- import("openai")
tiktoken <- import("tiktoken")
encoder <- tiktoken$encoding_for_model('gpt-4')

get_n_tokens <- function(txt) {
  txt |> unlist(use.names = FALSE) |> str_flatten_lines() |>
    encoder$encode() |>
    length()
}


get_cost <- function(completion, ..., model = completion$model,
                     prompt_tokens = completion$usage$prompt_tokens,
                     completion_tokens = completion$usage$completion_tokens
) {
  cost <-
       if(model |> startsWith("gpt-4-32k"))          list(input = 0.06,   output = 0.12)
  else if(model |> startsWith("gpt-4"))              list(input = 0.03,   output = 0.06)
  else if(model |> startsWith("gpt-3.5-turbo-16k"))  list(input = 0.003,  output = 0.004)
  else if(model |> startsWith("gpt-3.5-turbo"))      list(input = 0.0015, output = 0.002)
  cost$input * (prompt_tokens / 1000) + cost$output * (completion_tokens / 1000)
}


get_roxygen <- function(path) {
  x <- readLines(path)
  x <- x[startsWith(x, "#'")]
  str_flatten_lines(x)
}


get_translated_roxygen <- function(roxygen) {
  roxygen %<>% str_flatten_lines()
  if(!str_detect(roxygen, fixed("\n")))
    roxygen %<>% get_roxygen()

  (completion <- openai$ChatCompletion$create(
    # model="gpt-4",                #  8,192 context window
    # model="gpt-4-32k"             # 32,768 context window
    model="gpt-3.5-turbo",          #  4,097 context window
    # model="gpt-3.5-turbo-16k",    # 16,385 context window
    temperature = 0,
    max_tokens = as.integer(round(length(encoder$encode(roxygen))*1.2)),
    messages = chat_messages(
      !!!prompt_translate_roxygen_instructions_and_examples,
      user = roxygen
    ))) |> system.time() -> runtime

  completion$runtime <- runtime[["elapsed"]]

  roxygen2 <- completion$choices[[1L]]$message$content
  attr(roxygen2, "completion") <- completion
  attr(roxygen2, "cost") <- get_cost(completion)
  roxygen2
}






prompt_translate_roxygen_instructions_and_examples <- list %(% {

  system = str_squish(r"{
    You translate Python to R (including docs, code, idioms), correct typos,
    and output properly formatted roxygen.
    You always wrap `NULL`, `TRUE` and `FALSE` in backticks as needed.
    You translate python code chunks to R code chunks.
    You output Rmd and/or roxygen.
    You leave everthing else the same.
  }")

  # ---- initializer ----
  user = r"---(
    #' Initializer that generates tensors initialized to 0.
    #'
    #' @description
    #'
    #' # Examples
    #' ```python
    #' # Standalone usage:
    #' initializer = Zeros()
    #' values = initializer(shape=(2, 2))
    #' ```
    #'
    #' ```python
    #' # Usage in a Keras layer:
    #' initializer = Zeros()
    #' layer = Dense(units=3, kernel_initializer=initializer)
    #' ```
    #'
    #' @family initializer
    #' @export
    )---"
  assistant = r"---(
    #' Initializer that generates tensors initialized to 0.
    #'
    #' @description
    #'
    #' # Examples
    #'
    #' ```{r}
    #' # Standalone usage:
    #' initializer <- initializer_zeros()
    #' values <- initializer(shape = c(2, 2))
    #' ```
    #'
    #' ```{r}
    #' # Usage in a Keras layer:
    #' initializer <- initializer_zeros()
    #' layer <- layer_dense(units = 3, kernel_initializer = initializer)
    #' ```
    #'
    #' @family initializer
    #' @export
    )---"

  # ---- layer_embedding ----
  user = r"---(
    #' Turns positive integers (indexes) into dense vectors of fixed size.
    #'
    #' @description
    #' e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`
    #'
    #' This layer can only be used on positive integer inputs of a fixed range.
    #'
    #' # Examples
    #' ```python
    #' model = keras.Sequential()
    #' model.add(keras.layers.Embedding(1000, 64, input_length=10))
    #' # The model will take as input an integer matrix of size (batch,
    #' # input_length), and the largest integer (i.e. word index) in the input
    #' # should be no larger than 999 (vocabulary size).
    #' # Now model.output_shape is (None, 10, 64), where `None` is the batch
    #' # dimension.
    #' input_array = np.random.randint(1000, size=(32, 10))
    #' model.compile('rmsprop', 'mse')
    #' output_array = model.predict(input_array)
    #' print(output_array.shape)
    #' # (32, 10, 64)
    #' ```
    #'
    #' # Input Shape
    #' 2D tensor with shape: `(batch_size, input_length)`.
    #'
    #' # Output Shape
    #'     3D tensor with shape: `(batch_size, input_length, output_dim)`.
    #'
    #' @param input_dim Integer. Size of the vocabulary,
    #'     i.e. maximum integer index + 1.
    #' @param output_dim Integer. Dimension of the dense embedding.
    #' @param embeddings_initializer Initializer for the `embeddings`
    #'     matrix (see `keras.initializers`).
    #' @param embeddings_regularizer Regularizer function applied to
    #'     the `embeddings` matrix (see `keras.regularizers`).
    #' @param embeddings_constraint Constraint function applied to
    #'     the `embeddings` matrix (see `keras.constraints`).
    #' @param mask_zero Boolean, whether or not the input value 0 is a special
    #'     "padding" value that should be masked out.
    #'     This is useful when using recurrent layers which
    #'     may take variable length input. If this is `True`,
    #'     then all subsequent layers in the model need
    #'     to support masking or an exception will be raised.
    #'     If mask_zero is set to True, as a consequence,
    #'     index 0 cannot be used in the vocabulary (input_dim should
    #'     equal size of vocabulary + 1).
    #' @param object Object to compose the layer with. A tensor, array, or sequential model.
    #' @param ... Passed on to the Python callable
    #'
    #' @family core layers
    #' @export
    )---"
  assistant = r"---(
    #' Turns positive integers (indexes) into dense vectors of fixed size.
    #'
    #' @description
    #' e.g. `c(4, 20) -> rbind(c(0.25, 0.1), c(0.6, -0.2))`
    #'
    #' This layer can only be used on positive integer inputs of a fixed range.
    #'
    #' # Examples
    #' ```{r}
    #' model <- keras_model_sequential() %>%
    #'   layer_embedding(1000, 64, input_length = 10)
    #'
    #' # The model will take as input an integer matrix of size (batch,
    #' # input_length), and the largest integer (i.e. word index) in the input
    #' # should be no larger than 999 (vocabulary size).
    #' # Now model$output_shape is (NA, 10, 64), where `NA` is the batch
    #' # dimension.
    #'
    #' input_array <- random_array(c(32, 10), gen = sample.int)
    #' model %>% compile('rmsprop', 'mse')
    #' output_array <- model %>% predict(input_array)
    #' print(dim(output_array))   # (32, 10, 64)
    #' ```
    #'
    #' # Input Shape
    #'     2D tensor with shape: `(batch_size, input_length)`.
    #'
    #' # Output Shape
    #'     3D tensor with shape: `(batch_size, input_length, output_dim)`.
    #'
    #' @param input_dim Integer. Size of the vocabulary,
    #'     i.e. maximum integer index + 1.
    #' @param output_dim Integer. Dimension of the dense embedding.
    #' @param embeddings_initializer Initializer for the `embeddings`
    #'     matrix (see `keras::initializer_*`).
    #' @param embeddings_regularizer Regularizer function applied to
    #'     the `embeddings` matrix (see `keras::regularizer_*`).
    #' @param embeddings_constraint Constraint function applied to
    #'     the `embeddings` matrix (see `keras::constraint_*`).
    #' @param mask_zero Boolean, whether or not the input value 0 is a special
    #'     "padding" value that should be masked out.
    #'     This is useful when using recurrent layers which
    #'     may take variable length input. If this is `TRUE`,
    #'     then all subsequent layers in the model need
    #'     to support masking or an exception will be raised.
    #'     If `mask_zero` is set to `TRUE`, as a consequence,
    #'     index 0 cannot be used in the vocabulary (`input_dim` should
    #'     equal size of vocabulary + 1).
    #' @param object Object to compose the layer with. A tensor, array, or sequential model.
    #' @param ... For forward/backward compatability.
    #' @family core layers
    #' @export
    #' @examples

    )---"


  # ---- layer_conv_2d example ----
  user = r"--(
    #' This layer creates a convolution kernel that is convolved
    #' with the layer input to produce a tensor of
    #' outputs. If `use_bias` is True,
    #' a bias vector is created and added to the outputs. Finally, if
    #' `activation` is not `None`, it is applied to the outputs as well.
    )--"

  assistant = r"--(
    #' This layer creates a convolution kernel that is convolved
    #' with the layer input to produce a tensor of
    #' outputs. If `use_bias` is `TRUE`,
    #' a bias vector is created and added to the outputs. Finally, if
    #' `activation` is not `NULL`, it is applied to the outputs as well.
    )--"

  # ---- layer_conv_2d example ----
  user = r"--(
    #' ```python
    #' # The inputs are 28x28 RGB images with `channels_last` and the batch
    #' # size is 4.
    #' input_shape = (4, 28, 28, 3)
    #' x = tf.random.normal(input_shape)
    #' y = tf.keras.layers.Conv2D(
    #' 2, 3, activation='relu', input_shape=input_shape[1:])(x)
    #' print(y.shape)
    #' # (4, 26, 26, 2)
    #' ```
    )--"

  assistant = r"--(
    #' ```{r}
    #' # The inputs are 28x28 RGB images with `channels_last` and the batch
    #' # size is 4.
    #' input_shape <- shape(4, 28, 28, 3)
    #' x <- tf$random$normal(input_shape)
    #' y <- x |>
    #'   layer_conv_2d(2, 3, activation='relu')
    #' y$shape   # (4, 26, 26, 2)
    #' ```
    )--"

  # ---- layer_hashing example ----
  user = r"---(
    #' ```python
    #' layer = keras.layers.Hashing(num_bins=3, salt=133)
    #' inp = [['A'], ['B'], ['C'], ['D'], ['E']]
    #' layer(inp)
    #' # array([[0],
    #' #         [0],
    #' #         [2],
    #' #         [1],
    #' #         [0]])
    #' ```
    )---"

  assistant = r"---(
    #' ```{r}
    #' layer <- layer_hashing(num_bins = 3, salt = 133)
    #' inp <- c('A', 'B', 'C', 'D', 'E')
    #' layer(inp)
    #' ```
    )---"
}














# get_n_tokens.chat_messages <- function(x) {
#   # UseMethod("get_n_tokens")
#
# }
#
# get_n_tokens.character <- function(x) {
#
# }
