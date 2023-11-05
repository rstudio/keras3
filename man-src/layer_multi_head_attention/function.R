layer_multi_head_attention <-
function (inputs, num_heads, key_dim, value_dim = NULL, dropout = 0, 
    use_bias = TRUE, output_shape = NULL, attention_axes = NULL, 
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros", 
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL, 
    kernel_constraint = NULL, bias_constraint = NULL, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        num_heads = as_integer, key_dim = as_integer, value_dim = as_integer, 
        attention_axes = as_integer), ignore = "inputs")
    layer <- do.call(keras$layers$MultiHeadAttention, args)
    if (missing(inputs) || is.null(inputs)) 
        return(layer)
    if (!is.list(inputs)) 
        inputs <- list(inputs)
    do.call(layer, inputs)
}
