
if(!"source:tools/utils.R" %in% search())
  envir::attach_source("tools/utils.R")


# game plan:
#
# for each r_wrapper, write out:
# tools/raw
#   layer_dense/
#     docstring.txt                      - the raw docstring(s) for the python endpoints
#     r_wrapper_intermediate.R           - the manually transformed docstring to roxygen
#     r_wrapper.R                        - The final pass w/ first pass by llm + human massaging before checkin to git.
#                                          Only calls out to openai as needed from updates for the first pass- but up to the
#                                          user (me) to make sure it's correct, or to manually correct it, before checking it in.

# Two workflows to consider here:
#   Setup the initial scaffolding - generate the R wrapper, parse docstring to extract out
#   params for building the wrapper, etc.
#   Updates.
#
#   Can LLMs do the whole thing?
#   maybe the workflow is some kind of button to generate? The key is to limit the interaction w/ the LLM because it's
#   expensive, and also, it needs to be deliberate, managed with a specific prompt.
#
#   The alternative of course it to work with layers
#   the docstring + literal docstring2roxygen is one layer, and then llm modifications are patches atop that.
#   Then, patches can be edited by hand somehow? there needs to be a nice workflow for "fixing up" llm output,
#   but then in such a way that patches can easily be layered atop
#   e.g., 3.1 - we don't want to have to call out to llm for every change,
#   we want to see a diff where the updates are added to roxygen, and then
#   former "fixes" are applied atop (or if fails), then calls out to llm for a fresh
#   first pass.
#   Maybe, instead of a patch file, we instead dynamically generate the patch file
#   from files on the filesystem, then attempt to layer it atop.
#   So, sequence of updating a wrapper is
#   1. compare docstring in fs vs in live object. if not same, then:
#   2. take the old docstring
#     - write out a temporary intermediate r wrapper from the old docstring.
#     - compare it to the current r_wrapper of record and generate an intermediate
#       patch file that contains just the llm/manual changes.
#   2. write out new docstring
#   3. write out a new temporary intermediate r wrapper
#   4. attempt to apply the patch from step 2 to the new intermediate r wrapper
#     if successful, just replace the successfully applied patch. If not,
#     then resolve it as a regular git merge conflict.
#
# # TO IMPLEMENT:
# # Workflow Steps for Updating/Generating a Wrapper
# 1. Compare Current and Live Docstrings
#    Check if the docstring in the file system (fs) is the same as in the live
#    Python object. If they are the same, just reuse the previous wrapper.
#    If not the same, proceed to the next steps.

# 2. Generate Intermediate Patch File
#    - Use the old docstring to write out a temporary intermediate R wrapper
#      using programmatic parsing + code gen only.
#    - Compare this temporary R wrapper with the current R wrapper of record.
#    - Generate an intermediate "fixups.patch" file containing only changes made
#      post programmatic roxygen+wrapper autogen.
#      (that is, by an LLM or manually)

# 3. Write Out New Docstring
#    Save the new docstring from the live Python object to the file system.

# 4. Generate New Intermediate R Wrapper
#    Write out a new temporary intermediate R wrapper based on the new docstring.

# 5. Apply Patch
#    - Attempt to apply the patch generated in Step 2 to the new temporary
#      intermediate R wrapper.
#    - If successful, replace the R wrapper with the patched version.
#    - If it fails, resolve it as a regular Git merge conflict.


## What about tests?

gptrd_files <- Sys.glob("tools/raw/*/gpt-4*") %>%
  .[!duplicated(dirname(.))] %>%
  normalizePath()

invisible(lapply(gptrd_files, \(f) {
  withr::local_dir(dirname(f))
  roxygen <- readLines(f)
  wrapper <- readLines("r-wrapper-literal.R") %>% .[!startsWith(., "#")]
  writeLines(str_flatten_lines(roxygen, wrapper), "r-wrapper-llm.R")
}))

  # }
  # file.copy(., file.path(dirname(.), "r-wrapper-llm.R"),
            # overwrite = TRUE)


# py_install("openai")
use_codellama <- FALSE
if (use_codellama) {
  Sys.unsetenv("OPENAI_API_KEY")
  openai <- import("openai")
  openai$api_base <- glue("http://192.168.1.15:9092")

} else {
  readRenviron("~/.Renviron")
  import("os")$environ$update(list(OPENAI_API_KEY = Sys.getenv("OPENAI_API_KEY")))
  openai <- import("openai")
}



# TODO: bulleted list in image_dataset_from_directory in arg `label_mode` not formatting correctly
#
# if(FALSE) {
# models <- openai$Model$list()
#
# models[1:2] %>% str()
#
# map_chr(models$data, "id") %>%
#   grep("4", ., value = T)
# }
# py_install("tiktoken")

tiktoken <- import("tiktoken")
encoder <- tiktoken$encoding_for_model('gpt-3.5-turbo')
encoder$encode(get_roxygen(path)) |> length()

get_translation <- function(path) {

  roxygen <- get_roxygen(path)

time <- system.time(chat_completion <- openai$ChatCompletion$create(
  model="gpt-4",                #  8,192 context window
  # model="gpt-4-32k"             # 32,768 context window
  # model="gpt-3.5-turbo",        #  4,097 context window
  # model="gpt-3.5-turbo-16k",      # 16,385 context window
  temperature = 0,
  max_tokens = as.integer(round(length(encoder$encode(roxygen))*1.1)),
  messages = chat_messages %(% {

    system = str_squish(
              "You translate all Python idioms and code to R, correct typos,
              wrap `NULL`, `TRUE` and `FALSE` in backticks as needed for properly
              formatted roxygen,
              move examples from description to an @examples roxygen section
              leave everthing else the same.
               ")

    ###
    # user = "x = tf.random.normal(input_shape)"
    # assistant = "x <- tf$random$normal(input_shape)"

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
    #' @export
    #' @family initializer
    #' @seealso
    #' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/zeros>
    )---"

    assistant = r"---(
    #' Initializer that generates tensors initialized to 0.
    #'
    #' @export
    #' @family initializer
    #' @seealso
    #' + <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/zeros>
    #'
    #' @examples
    #' # Standalone usage:
    #' initializer <- initializer_zeros()
    #' values <- initializer(shape = c(2, 2))
    #'
    #' # Usage in a Keras layer:
    #' initializer <- initializer_zeros()
    #' layer <- layer_dense(units = 3, kernel_initializer = initializer)
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
#' @export
#' @family core layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding>
    )---"


    assistant = r"---(
#' Turns positive integers (indexes) into dense vectors of fixed size.
#'
#' @description
#' e.g. `c(4, 20) -> rbind(c(0.25, 0.1), c(0.6, -0.2))`
#'
#' This layer can only be used on positive integer inputs of a fixed range.
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
#'     index 0 cannot be used in the vocabulary (input_dim should
#'     equal size of vocabulary + 1).
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#' @export
#' @family core layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding>
#'
#' @examples
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
#' print(dim(output_array))
#' # (32, 10, 64)
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
#' @examples
#' # The inputs are 28x28 RGB images with `channels_last` and the batch
#' # size is 4.
#' input_shape = shape(4, 28, 28, 3)
#' x <- tf$random$normal(input_shape)
#' y <- x |>
#'   layer_conv_2d(2, 3, activation='relu')
#' y$shape   # (4, 26, 26, 2)
)--"

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
#' @examples
#' layer <- layer_hashing(num_bins = 3, salt = 133)
#' inp <- c('A', 'B', 'C', 'D', 'E')
#' layer(inp)
)---"

    ## maybe we leave tuple refs in the docs, reexport reticulate::tuple()
#      user = r"---(
#  #' @param salt A single unsigned integer or None.
#  #'     If passed, the hash function used will be SipHash64,
#  #'     with these values used as an additional input
#  #'     (known as a "salt" in cryptography).
#  #'     These should be non-zero. If `None`, uses the FarmHash64 hash
#  #'     function. It also supports tuple/list of 2 unsigned
#  #'     integer numbers, see reference paper for details.
#  #'     Defaults to `None`.
#      )---"
#      assistant = r"---(
#  #' @param salt A single unsigned integer or `NULL`.
#  #'     If passed, the hash function used will be SipHash64,
#  #'     with these values used as an additional input
#  #'     (known as a "salt" in cryptography).
#  #'     These should be non-zero. If `NULL`, uses the FarmHash64 hash
#  #'     function. It also supports a list of 2 unsigned
#  #'     integer numbers, see reference paper for details.
#  #'     Defaults to `NULL`.
#  )---"

    user = get_roxygen(path)
  }
))
print(time)
chat_completion
}

get_roxygen <- function(path) {
  path %>% readLines() %>% .[startsWith(., "#'")] %>%
    str_flatten_lines()
}

lapply(df)

path <- "tools/raw/keras-layers-Hashing/r_wrapper.R"
chat_completion <- get_translation(path)
writeLines(chat_completion$choices[[1]]$message$content,
           file.path(
             dirname(path),
             sprintf("%s.txt", chat_completion$model)))

stop("DONE")

chat_completion$usage$prompt_tokens


# rate limits:
# https://platform.openai.com/account/rate-limits

# gpt4 pricing
# Model	      Input	            Output
# 8K context	$0.03 / 1K tokens	$0.06 / 1K tokens
# 32K context	$0.06 / 1K tokens	$0.12 / 1K tokens
#
# gpt3.5
# Model	      Input	              Output
# 4K context	$0.0015 / 1K tokens	$0.002 / 1K tokens
# 16K context	$0.003 / 1K tokens	$0.004 / 1K tokens

# nchar(get_roxygen(path)), chat_completion$usage
#                              $prompt_tokens,
#                                $completion_tokens,
#                                   $total_tokens    elapsed  time (sec)  cost
# layer_hashing()   5279,  3380, 1287, 4667          106.039               0.17
#
# gpt3.516k 184.815 secs

get_cost <- function(completion, ..., model = completion$model,
                     prompt_tokens = completion$usage$prompt_tokens,
                     completion_tokens = completion$usage$completion_tokens
                     ) {
  cost <-
       if(model |> startsWith("gpt-4-32k"))          list(input = 0.06, output = 0.12)
  else if(model |> startsWith("gpt-4"))              list(input = 0.03, output = 0.06)
  else if(model |> startsWith("gpt-3.5-turbo-16k"))  list(input = 0.003, output = 0.004)
  else if(model |> startsWith("gpt-3.5-turbo"))      list(input = 0.0015, output = 0.002)
  cost$input * (prompt_tokens / 1000) + cost$output * (completion_tokens / 1000)
}

get_cost(chat_completion)                               # 0.17
get_cost(chat_completion, model = "gpt-4-32k")          # 0.35
get_cost(chat_completion, model = "gpt-3.5-turbo")      # 0.007
get_cost(chat_completion, model = "gpt-3.5-turbo-16k")  # 0.015

str(chat_completion)

chat_completion$choices[[1]]$message$content |> cat()

if(FALSE) {
# source("tools/make.R")
df %>%
  mutate(doc_len = str_length(docstring)) %>%
  # arrange(doc_len) %>%
  arrange(-doc_len) %>%
  filter(doc_len > 100) %>%
  filter(str_detect(dump, "```python")) %>%
  select(r_name, endpoint, docstring, doc_len, dump) %>%
  mutate(dump_tokens = map_int(dump, ~length(encoder$encode(.x)))) %>%
  arrange((dump_tokens)) %>%
  pull(endpoint) %>%
  str_replace_all(fixed("."), "-") %>%
  TKutils::dput_cb()
}

# 120 seconds w/ Conv2D for code llama
# 106 seconds w/ Conv2D for openai





waldo::compare(
  get_roxygen(path),
  chat_completion$choices[[1]]$message$content
)

get_roxygen(path) %>% cat()





c("keras-ops-cast", "keras-ops-arccosh", "keras-ops-convert_to_tensor",
  "keras-ops-stop_gradient", "keras-initializers-zeros", "keras-ops-arcsinh",
  "keras-ops-absolute", "keras-initializers-ones", "keras-ops-rsqrt",
  "keras-ops-arctan", "keras-initializers-identity", "keras-ops-arcsin",
  "keras-ops-array", "keras-regularizers-l2", "keras-regularizers-l1",
  "keras-ops-relu", "keras-ops-erf", "keras-initializers-constant",
  "keras-ops-arccos", "keras-ops-softplus", "keras-ops-log_sigmoid",
  "keras-ops-softsign", "keras-ops-sigmoid", "keras-ops-fft", "keras-ops-shape",
  "keras-callbacks-CSVLogger", "keras-ops-relu6", "keras-ops-broadcast_to",
  "keras-ops-selu", "keras-ops-digitize", "keras-ops-hard_sigmoid",
  "keras-ops-unstack", "keras-ops-elu", "keras-ops-log_softmax",
  "keras-regularizers-L1L2", "keras-ops-fori_loop", "keras-ops-fft2",
  "keras-layers-RepeatVector", "keras-ops-logsumexp", "keras-ops-leaky_relu",
  "keras-ops-scatter", "keras-ops-slice", "keras-layers-StackedRNNCells",
  "keras-ops-qr", "keras-ops-add", "keras-ops-silu", "keras-ops-count_nonzero",
  "keras-metrics-MeanMetricWrapper", "keras-ops-top_k", "keras-ops-argmax",
  "keras-preprocessing-image-img_to_array", "keras-ops-softmax",
  "keras-ops-argmin", "keras-layers-Activation", "keras-layers-UnitNormalization",
  "keras-ops-in_top_k", "keras-ops-gelu", "keras-metrics-Sum",
  "keras-metrics-kl_divergence", "keras-ops-multi_hot", "keras-metrics-Mean",
  "keras-ops-segment_sum", "keras-ops-while_loop", "keras-ops-segment_max",
  "keras-metrics-huber", "keras-ops-extract_sequences", "keras-layers-Permute",
  "keras-initializers-HeNormal", "keras-preprocessing-image-array_to_img",
  "keras-initializers-HeUniform", "keras-initializers-LecunUniform",
  "keras-metrics-log_cosh", "keras-initializers-RandomNormal",
  "keras-metrics-RootMeanSquaredError", "keras-layers-Flatten",
  "keras-metrics-LogCoshError", "keras-ops-moments", "keras-ops-argsort",
  "keras-initializers-GlorotUniform", "keras-layers-Reshape", "keras-initializers-GlorotNormal",
  "keras-initializers-RandomUniform", "keras-regularizers-OrthogonalRegularizer",
  "keras-layers-Softmax", "keras-initializers-LecunNormal", "keras-ops-meshgrid",
  "keras-initializers-TruncatedNormal", "keras-ops-binary_crossentropy",
  "keras-initializers-orthogonal", "keras-ops-diag", "keras-losses-MeanAbsoluteError",
  "keras-preprocessing-image-load_img", "keras-losses-MeanSquaredError",
  "keras-callbacks-LearningRateScheduler", "keras-ops-slice_update",
  "keras-ops-all", "keras-ops-any", "keras-ops-amin", "keras-ops-amax",
  "keras-metrics-MeanSquaredError", "keras-layers-Add", "keras-layers-Multiply",
  "keras-layers-Maximum", "keras-layers-Minimum", "keras-layers-Average",
  "keras-layers-Concatenate", "keras-ops-rfft", "keras-ops-one_hot",
  "keras-layers-Subtract", "keras-layers-SpectralNormalization",
  "keras-activations-relu", "keras-layers-Cropping1D", "keras-metrics-SparseTopKCategoricalAccuracy",
  "keras-losses-Poisson", "keras-losses-MeanAbsolutePercentageError",
  "keras-metrics-FalseNegatives", "keras-metrics-FalsePositives",
  "keras-metrics-TrueNegatives", "keras-losses-KLDivergence", "keras-metrics-TruePositives",
  "keras-layers-UpSampling1D", "keras-ops-append", "keras-metrics-TopKCategoricalAccuracy",
  "keras-losses-Hinge", "keras-ops-sparse_categorical_crossentropy",
  "keras-ops-categorical_crossentropy", "keras-metrics-BinaryAccuracy",
  "keras-losses-MeanSquaredLogarithmicError", "keras-losses-SquaredHinge",
  "keras-ops-bincount", "keras-metrics-MeanAbsoluteError", "keras-losses-CategoricalHinge",
  "keras-ops-istft", "keras-losses-LogCosh", "keras-callbacks-ReduceLROnPlateau",
  "keras-layers-Masking", "keras-ops-diagonal", "keras-ops-arctan2",
  "keras-ops-stft", "keras-initializers-VarianceScaling", "keras-ops-irfft",
  "keras-metrics-categorical_focal_crossentropy", "keras-layers-TimeDistributed",
  "keras-losses-Huber", "keras-metrics-SparseCategoricalAccuracy",
  "keras-layers-ZeroPadding1D", "keras-metrics-Hinge", "keras-metrics-R2Score",
  "keras-layers-TFSMLayer", "keras-metrics-CategoricalHinge", "keras-layers-Input",
  "keras-metrics-CosineSimilarity", "keras-metrics-SquaredHinge",
  "keras-layers-GlobalMaxPooling1D", "keras-metrics-Poisson", "keras-metrics-CategoricalAccuracy",
  "keras-metrics-MeanAbsolutePercentageError", "keras-ops-average",
  "keras-layers-GlobalMaxPooling2D", "keras-layers-GlobalAveragePooling2D",
  "keras-ops-arange", "keras-metrics-KLDivergence", "keras-optimizers-schedules-PiecewiseConstantDecay",
  "keras-layers-GlobalAveragePooling1D", "keras-metrics-MeanSquaredLogarithmicError",
  "keras-metrics-PrecisionAtRecall", "keras-metrics-F1Score", "keras-optimizers-schedules-ExponentialDecay",
  "keras-layers-Lambda", "keras-metrics-RecallAtPrecision", "keras-optimizers-schedules-InverseTimeDecay",
  "keras-layers-GlobalMaxPooling3D", "keras-layers-GlobalAveragePooling3D",
  "keras-layers-UpSampling3D", "keras-losses-CosineSimilarity",
  "keras-callbacks-LambdaCallback", "keras-layers-Dot", "keras-layers-TorchModuleWrapper",
  "keras-optimizers-schedules-CosineDecayRestarts", "keras-layers-Cropping2D",
  "keras-layers-Embedding", "keras-metrics-FBetaScore", "keras-ops-scatter_update",
  "keras-metrics-SensitivityAtSpecificity", "keras-metrics-BinaryCrossentropy",
  "keras-metrics-SpecificityAtSensitivity", "keras-preprocessing-sequence-pad_sequences",
  "keras-metrics-binary_focal_crossentropy", "keras-layers-ZeroPadding3D",
  "keras-layers-ZeroPadding2D", "keras-layers-Cropping3D", "keras-layers-UpSampling2D",
  "keras-callbacks-BackupAndRestore", "keras-metrics-Recall", "keras-metrics-MeanIoU",
  "keras-layers-AveragePooling3D", "keras-callbacks-EarlyStopping",
  "keras-optimizers-schedules-PolynomialDecay", "keras-layers-MaxPooling3D",
  "keras-layers-HashedCrossing", "keras-layers-SimpleRNNCell",
  "keras-metrics-BinaryIoU", "keras-layers-RandomBrightness", "keras-ops-einsum",
  "keras-losses-CategoricalCrossentropy", "keras-layers-AveragePooling1D",
  "keras-metrics-CategoricalCrossentropy", "keras-layers-MaxPooling1D",
  "keras-layers-Bidirectional", "keras-optimizers-SGD", "keras-losses-SparseCategoricalCrossentropy",
  "keras-layers-GRUCell", "keras-metrics-SparseCategoricalCrossentropy",
  "keras-metrics-IoU", "keras-metrics-Precision", "keras-layers-LSTMCell",
  "keras-metrics-OneHotMeanIoU", "keras-layers-EinsumDense", "keras-optimizers-schedules-CosineDecay",
  "keras-layers-Discretization", "keras-layers-CategoryEncoding",
  "keras-metrics-OneHotIoU", "keras-optimizers-Adamax", "keras-layers-Normalization",
  "keras-layers-AveragePooling2D", "keras-layers-MaxPooling2D",
  "keras-optimizers-RMSprop", "keras-layers-Conv1DTranspose", "keras-losses-BinaryCrossentropy",
  "keras-layers-SimpleRNN", "keras-layers-LayerNormalization",
  "keras-layers-Conv2DTranspose", "keras-layers-RandomZoom", "keras-layers-Conv2D",
  "keras-layers-DepthwiseConv1D", "keras-layers-DepthwiseConv2D",
  "keras-layers-Conv3DTranspose", "keras-preprocessing-timeseries_dataset_from_array",
  "keras-layers-SeparableConv1D", "keras-layers-Conv3D", "keras-layers-Conv1D",
  "keras-layers-SeparableConv2D", "keras-optimizers-Ftrl", "keras-callbacks-ModelCheckpoint",
  "keras-layers-Hashing", "keras-callbacks-TensorBoard", "keras-losses-CategoricalFocalCrossentropy",
  "keras-layers-LSTM", "keras-layers-GRU", "keras-metrics-AUC",
  "keras-layers-RNN", "keras-losses-BinaryFocalCrossentropy", "keras-layers-TextVectorization",
  "keras-layers-StringLookup", "keras-layers-IntegerLookup") -> endpoints


total_spent <- 0
for(e in endpoints) {
  withr::with_dir(file.path("~/github/rstudio/keras/tools/raw/", e), {
    message(e)
    chat_completion <- get_translation("r_wrapper.R")
    writeLines(chat_completion$choices[[1]]$message$content,
               sprintf("%s.txt", chat_completion$model))
    str(chat_completion)
    cat(chat_completion$choices[[1]]$message$content)
    cat("\n")
    total_spent <- total_spent + get_cost(chat_completion)
    message("total spent: ", total_spent)
    print(Sys.time())
    if(total_spent > 5)
      break
  })
}


#     user = r"---(
#     #' Scaled Exponential Linear Unit (SELU).
#     #'
#     #' @description
#     #' The Scaled Exponential Linear Unit (SELU) activation function is defined as:
#     #'
#     #' - `scale * x` if `x > 0`
#     #' - `scale * alpha * (exp(x) - 1)` if `x < 0`
#     #'
#     #' where `alpha` and `scale` are pre-defined constants
#     #' (`alpha=1.67326324` and `scale=1.05070098`).
#     #'
#     #' Basically, the SELU activation function multiplies `scale` (> 1) with the
#     #' output of the `keras.activations.elu` function to ensure a slope larger
#     #' than one for positive inputs.
#     #'
#     #' The values of `alpha` and `scale` are
#     #' chosen so that the mean and variance of the inputs are preserved
#     #' between two consecutive layers as long as the weights are initialized
#     #' correctly (see `keras.initializers.LecunNormal` initializer)
#     #' and the number of input units is "large enough"
#     #' (see reference paper for more information).
#     #'
#     #' # Notes
#     #' - To be used together with the
#     #'     `keras.initializers.LecunNormal` initializer.
#     #' - To be used together with the dropout variant
#     #'     `keras.layers.AlphaDropout` (rather than regular dropout).
#     #'
#     #' # Reference
#     #' - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
#     #'
#     #' @param x Input tensor.
#     #'
#     #' @export
#     #' @family activation functions
#     #' @seealso
#     #' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>
#     )---"
#
#     assistant = r"---(
#     #' Scaled Exponential Linear Unit (SELU).
#     #'
#     #' @description
#     #' The Scaled Exponential Linear Unit (SELU) activation function is defined as:
#     #'
#     #' - `scale * x` if `x > 0`
#     #' - `scale * alpha * (exp(x) - 1)` if `x < 0`
#     #'
#     #' where `alpha` and `scale` are pre-defined constants
#     #' (`alpha=1.67326324` and `scale=1.05070098`).
#     #'
#     #' Basically, the SELU activation function multiplies `scale` (> 1) with the
#     #' output of the [`activation_elu()`] function to ensure a slope larger
#     #' than one for positive inputs.
#     #'
#     #' The values of `alpha` and `scale` are
#     #' chosen so that the mean and variance of the inputs are preserved
#     #' between two consecutive layers as long as the weights are initialized
#     #' correctly (see [`initializer_lecun_normal()`])
#     #' and the number of input units is "large enough"
#     #' (see reference paper for more information).
#     #'
#     #' # Notes
#     #' - To be used together with the
#     #'     [`initializer_lecun_normal()`] initializer.
#     #' - To be used together with the dropout variant
#     #'     `keras.layers.AlphaDropout` (rather than regular dropout).
#     #'
#     #' # Reference
#     #' - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
#     #'
#     #' @param x Input tensor.
#     #'
#     #' @export
#     #' @family activation functions
#     #' @seealso
#     #' + <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>
#     )---"
###

