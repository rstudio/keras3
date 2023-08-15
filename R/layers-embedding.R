

#' Turns positive integers (indexes) into dense vectors of fixed size
#'
#' @details
#' For example, `list(4L, 20L) -> list(c(0.25, 0.1), c(0.6, -0.2))`.
#'
#' This layer can only be used on positive integer inputs of a fixed range. The
#' `layer_text_vectorization()`, `layer_string_lookup()`,
#' and `layer_integer_lookup()` preprocessing layers can help prepare
#' inputs for an `Embedding` layer.
#'
#' This layer accepts `tf.Tensor`, `tf.RaggedTensor` and `tf.SparseTensor`
#' input.
#'
#' @param object Layer or Model object
#'
#' @param input_dim Integer. Size of the vocabulary,
#' i.e. maximum integer index + 1.
#'
#' @param output_dim Integer. Dimension of the dense embedding.
#'
#' @param embeddings_initializer Initializer for the `embeddings`
#' matrix (see `keras.initializers`).
#'
#' @param embeddings_regularizer,activity_regularizer Regularizer function applied to
#' the `embeddings` matrix or to the activations (see `keras.regularizers`).
#'
#' @param embeddings_constraint Constraint function applied to
#' the `embeddings` matrix (see `keras.constraints`).
#'
#' @param mask_zero Boolean, whether or not the input value 0 is a special
#' "padding" value that should be masked out. This is useful when using
#' recurrent layers which may take variable length input. If this is
#' `TRUE`, then all subsequent layers in the model need to support masking
#' or an exception will be raised. If mask_zero is set to TRUE, as a
#' consequence, index 0 cannot be used in the vocabulary (input_dim should
#' equal size of vocabulary + 1).
#'
#' @param input_length Length of input sequences, when it is constant.
#' This argument is required if you are going to connect
#' `Flatten` then `Dense` layers upstream
#' (without it, the shape of the dense outputs cannot be computed).
#'
#' @param sparse If TRUE, calling this layer returns a `tf.SparseTensor`. If FALSE,
#' the layer returns a dense `tf.Tensor`. For an entry with no features in
#' a sparse tensor (entry with value 0), the embedding vector of index 0 is
#' returned by default.
#'
#' @param ... standard layer arguments.
#'
#' @section Input shape: 2D tensor with shape: `(batch_size, sequence_length)`.
#'
#' @section Output shape: 3D tensor with shape: `(batch_size, sequence_length,
#'   output_dim)`.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding>
#'   +  <https://keras.io/api/layers>
#' @export
layer_embedding <-
  function(object, input_dim, output_dim, embeddings_initializer = "uniform",
           embeddings_regularizer = NULL, activity_regularizer = NULL,
           embeddings_constraint = NULL, mask_zero = FALSE, input_length = NULL,
           sparse = FALSE, ...)
  {
    args <- capture_args(match.call(), list(
      input_dim = as.integer,
      output_dim = as.integer,
      input_length = as_nullable_integer,
      batch_size = as_nullable_integer
    ), ignore = "object")
    create_layer(keras$layers$Embedding, object, args)
  }

