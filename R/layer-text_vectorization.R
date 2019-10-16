#' Text vectorization layer
#' 
#' This layer has basic options for managing text in a Keras model. It
#' transforms a batch of strings (one sample = one string) into either a list of
#' token indices (one sample = 1D tensor of integer token indices) or a dense
#' representation (one sample = 1D tensor of float values representing data about
#' the sample's tokens).
#' 
#' The processing of each sample contains the following steps:
#' 
#' 1) standardize each sample (usually lowercasing + punctuation stripping)
#' 2) split each sample into substrings (usually words)
#' 3) recombine substrings into tokens (usually ngrams)
#' 4) index tokens (associate a unique int value with each token)
#' 5) transform each sample using this index, either into a vector of ints or
#'    a dense float vector.
#'    
#' @inheritParams layer_dense
#' @param max_tokens The maximum size of the vocabulary for this layer. If `NULL`,
#'  there is no cap on the size of the vocabulary.
#' @param standardize Optional specification for standardization to apply to the
#'  input text. Values can be `NULL` (no standardization),
#'  `"lower_and_strip_punctuation"` (lowercase and remove punctuation) or a
#'  Callable. Default is `"lower_and_strip_punctuation"`.
#' @param split Optional specification for splitting the input text. Values can be
#'  `NULL` (no splitting), `"split_on_whitespace"` (split on ASCII whitespace), or 
#'  a Callable. Default is `"split_on_whitespace"`.
#' @param ngrams Optional specification for ngrams to create from the possibly-split
#'  input text. Values can be `NULL`, an integer or a list of integers; passing
#'  an integer will create ngrams up to that integer, and passing a list of
#'  integers will create ngrams for the specified values in the list. Passing
#'  `NULL` means that no ngrams will be created.
#' @param output_mode Optional specification for the output of the layer. Values can
#'  be `"int"`, `"binary"`, `"count"` or `"tfidf`, which control the outputs as follows:
#'  * "int": Outputs integer indices, one integer index per split string token.
#'  * "binary": Outputs a single int array per batch, of either vocab_size or
#'   `max_tokens` size, containing 1s in all elements where the token mapped
#'   to that index exists at least once in the batch item.
#'  * "count": As "binary", but the int array contains a count of the number of
#'   times the token at that index appeared in the batch item.
#'  * "tfidf": As "binary", but the TF-IDF algorithm is applied to find the value
#'   in each token slot.
#' @param output_sequence_lenght Only valid in "int" mode. If set, the output will have
#'  its time dimension padded or truncated to exactly `output_sequence_length`
#'  values, resulting in a tensor of shape (batch_size, output_sequence_length) regardless 
#'  of how many tokens resulted from the splitting step. Defaults to `NULL`.
#' @param pad_to_max_tokens Only valid in "binary", "count", and "tfidf" modes. If `TRUE`,
#'  the output will have its feature axis padded to `max_tokens` even if the
#'  number of unique tokens in the vocabulary is less than max_tokens,
#'  resulting in a tensor of shape (batch_size, max_tokens) regardless of
#'  vocabulary size. Defaults to `TRUE`.
#'  
#' @export
layer_text_vectorization <- function(object, max_tokens = NULL, standardize = "lower_and_strip_punctuation",
                                     split = "whitespace", ngrams = NULL, 
                                     output_mode = c("int", "binary", "count", "tfidf"),
                                     output_sequence_length = NULL, pad_to_max_tokens = TRUE,
                                     ...) {
  
  if (tensorflow::tf_version() < "2.1")
    stop("Text Vectorization requires TensorFlow version >= 2.1", call. = FALSE)
  
  if (length(ngrams) > 1)
    ngrams <- as_integer_tuple(ngrams)
  else
    ngrams <- as_nullable_integer(ngrams)
  
  output_mode <- match.arg(output_mode)
  
  create_layer(keras$layers$preprocessing$text_vectorization$TextVectorization, object, list(
    max_tokens = as_nullable_integer(max_tokens),
    standardize = resolve_global_var_arg(standardize),
    split = resolve_global_var_arg(split),
    ngrams = ngrams,
    output_mode = resolve_global_var_arg(output_mode),
    output_sequence_length = as_nullable_integer(output_sequence_length),
    pad_to_max_tokens = pad_to_max_tokens
  ))
}

resolve_global_var_arg <- function(x) {
  
  if (is.null(x))
    return(x)
  
  # because of `is` in this line: https://github.com/tensorflow/tensorflow/blob/cf43d6e7e41a4c4474f1056beb7d7b81bc8e9771/tensorflow/python/keras/layers/preprocessing/text_vectorization.py#L433
  # we need to point to exactly the same object
  tv <- import(
    paste0(resolve_implementation_module(), ".layers.preprocessing.text_vectorization"), 
    convert = FALSE
  )
  
  if (x == "lower_and_strip_punctuation") {
    x <- tv$LOWER_AND_STRIP_PUNCTUATION
  } else if (x == "whitespace") {
    x <- tv$SPLIT_ON_WHITESPACE
  } else if (x == "tfidf") {
    x <- tv$TFIDF
  } else if (x == "int") {
    x <- tv$INT  
  } else if (x == "binary") {
    x <- tv$BINARY
  } else if (x == "count") {
    x <- tv$COUNT
  }
  
  x
}

