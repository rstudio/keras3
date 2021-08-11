
#' Creates attention layer
#'
#' Dot-product attention layer, a.k.a. Luong-style attention.
#'
#' @inheritParams layer_dense
#'
#' @param inputs a list of inputs first should be the query tensor, the second the value tensor
#' @param use_scale If True, will create a scalar variable to scale the attention scores.
#' @param causal Boolean. Set to True for decoder self-attention. Adds a mask such that position i cannot attend to positions j > i.
#' This prevents the flow of information from the future towards the past.
#'
#' @family core layers
#' @family attention layers
#'
#' @export
layer_attention <- function(inputs,use_scale=FALSE, causal = FALSE, batch_size = NULL, dtype = NULL,
                                name = NULL, trainable = NULL, weights = NULL) {
  if (!is_tensorflow_implementation() || !tensorflow::tf_version() >= "1.14")
    stop("layer_dense_features requires TensorFlow implementation and version >= 1.14")
  create_layer(keras$layers$Attention, inputs, list(
      use_scale = use_scale,
      causal = causal,
      batch_size = batch_size,
      dtype = dtype,
      name = name,
      trainable = trainable,
      weights = weights)
    )


}

#' MultiHeadAttention layer
#'
#' This is an implementation of multi-headed attention based on "Attention is all
#' you Need". If query, key, value are the same, then this is self-attention.
#' Each timestep in query attends to the corresponding sequence in key, and returns
#' a fixed-width vector.
#'
#' This layer first projects query, key and value. These are (effectively) a list
#' of tensors of length num_attention_heads, where the corresponding shapes are
#' `[batch_size, , key_dim]`, `[batch_size, , key_dim]`, `[batch_size, , value_dim]`.
#'
#' Then, the query and key tensors are dot-producted and scaled. These are softmaxed
#' to obtain attention probabilities. The value tensors are then interpolated by
#' these probabilities, then concatenated back to a single tensor.
#'
#' Finally, the result tensor with the last dimension as value_dim can take an
#' linear projection and return.
#'
#' @inheritParams layer_attention
#' @param num_heads Number of attention heads.
#' @param key_dim Size of each attention head for query and key.
#' @param value_dim Size of each attention head for value.
#' @param dropout Dropout probability.
#' @param use_bias Boolean, whether the dense layers use bias vectors/matrices.
#' @param output_shape The expected shape of an output tensor, besides the batch and sequence dims. If not specified, projects back to the key feature dim.
#' @param attention_axes axes over which the attention is applied. None means attention over all axes, but batch, heads, and features.
#' @param kernel_initializer Initializer for dense layer kernels.
#' @param bias_initializer Initializer for dense layer biases.
#' @param kernel_regularizer Regularizer for dense layer kernels.
#' @param bias_regularizer Regularizer for dense layer biases.
#' @param activity_regularizer Regularizer for dense layer activity.
#' @param kernel_constraint Constraint for dense layer kernels.
#' @param bias_constraint Constraint for dense layer kernels.
#' @param ... Other arguments passed to the layer. Eg, `name`, `training`.
#'
#' @section Call arguments:
#'
#' * query: Query Tensor of shape `[B, T, dim]`.
#' * value: Value Tensor of shape `[B, S, dim]`.
#' * key: Optional key Tensor of shape `[B, S, dim]`. If not given, will use value
#'   for both key and value, which is the most common case.
#' * attention_mask: a boolean mask of shape `[B, T, S]`, that prevents attention
#'   to certain positions.
#' * return_attention_scores: A boolean to indicate whether the output should be
#'   attention output if TRUE, or (attention_output, attention_scores) if FALSE.
#'   Defaults to FALSE.
#' * training: Python boolean indicating whether the layer should behave in
#'   training mode (adding dropout) or in inference mode (no dropout). Defaults
#'   to either using the training mode of the parent layer/model, or FALSE
#'   (inference) if there is no parent layer.
#'
#' @return
#' - attention_output: The result of the computation, of shape `[B, T, E]`, where
#'   T is for target sequence shapes and E is the query input last dimension if
#'   output_shape is None. Otherwise, the multi-head outputs are project to the
#'   shape specified by output_shape.
#' - attention_scores: (Optional) multi-head attention coeffients over attention axes.
#'
#' @export
layer_multi_head_attention <- function(
  inputs,
  num_heads,
  key_dim,
  value_dim=NULL,
  dropout=0.0,
  use_bias=TRUE,
  output_shape=NULL,
  attention_axes=NULL,
  kernel_initializer="glorot_uniform",
  bias_initializer="zeros",
  kernel_regularizer=NULL,
  bias_regularizer=NULL,
  activity_regularizer=NULL,
  kernel_constraint=NULL,
  bias_constraint=NULL,
  ...
) {

  if (tensorflow::tf_version() < "2.4")
    stop("layer_multi_head_attention requires tf_version() >= 2.4")

  create_layer(keras$layers$MultiHeadAttention, inputs, list(
    num_heads=as.integer(num_heads),
    key_dim=as.integer(key_dim),
    value_dim=as.integer(value_dim),
    dropout=dropout,
    use_bias=use_bias,
    output_shape=output_shape,
    attention_axes=attention_axes,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
    kernel_constraint=kernel_constraint,
    bias_constraint=bias_constraint,
    ...
  ))
}

# TODO: finish + document: https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention
layer_additive_attention <- function(object, use_scale=TRUE, ...) {
  args <- capture_args(match.call())
  create_layer(keras$layers$AdditiveAttention, object, args)
}
