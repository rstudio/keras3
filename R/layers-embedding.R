

#' Turns positive integers (indexes) into dense vectors of fixed size.
#'
#' For example, `list(4L, 20L) -> list(c(0.25, 0.1), c(0.6, -0.2))` This layer
#' can only be used as the first layer in a model.
#'
#' @inheritParams layer_dense
#'
#' @param input_dim int > 0. Size of the vocabulary, i.e. maximum integer
#'   index + 1.
#' @param output_dim int >= 0. Dimension of the dense embedding.
#' @param embeddings_initializer Initializer for the `embeddings` matrix.
#' @param embeddings_regularizer Regularizer function applied to the
#'   `embeddings` matrix.
#' @param activity_regularizer activity_regularizer
#' @param embeddings_constraint Constraint function applied to the `embeddings`
#'   matrix.
#' @param mask_zero Whether or not the input value 0 is a special "padding"
#'   value that should be masked out. This is useful when using recurrent
#'   layers, which may take variable length inputs. If this is `TRUE` then all
#'   subsequent layers in the model need to support masking or an exception will
#'   be raised. If mask_zero is set to TRUE, as a consequence, index 0 cannot be
#'   used in the vocabulary (input_dim should equal size of vocabulary + 1).
#' @param input_length Length of input sequences, when it is constant. This
#'   argument is required if you are going to connect `Flatten` then `Dense`
#'   layers upstream (without it, the shape of the dense outputs cannot be
#'   computed).
#'
#' @section Input shape: 2D tensor with shape: `(batch_size, sequence_length)`.
#'
#' @section Output shape: 3D tensor with shape: `(batch_size, sequence_length,
#'   output_dim)`.
#'
#' @section References:
#' - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
#'
#' @export
layer_embedding <- function(object, input_dim, output_dim, embeddings_initializer = "uniform", embeddings_regularizer = NULL,
                            activity_regularizer = NULL, embeddings_constraint = NULL, mask_zero = FALSE, input_length = NULL,
                            batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$Embedding, object, list(
    input_dim = as.integer(input_dim),
    output_dim = as.integer(output_dim),
    embeddings_initializer = embeddings_initializer,
    embeddings_regularizer = embeddings_regularizer,
    activity_regularizer = activity_regularizer,
    embeddings_constraint = embeddings_constraint,
    mask_zero = mask_zero,
    input_length = if (!is.null(input_length)) as.integer(input_length) else NULL,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
}
