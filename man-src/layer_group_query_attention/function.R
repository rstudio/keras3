layer_group_query_attention <-
function (object, head_dim, num_query_heads, num_key_value_heads, 
    dropout = 0, use_bias = TRUE, kernel_initializer = "glorot_uniform", 
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL, 
    activity_regularizer = NULL, kernel_constraint = NULL, bias_constraint = NULL, 
    ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$GroupQueryAttention, object, args)
}
