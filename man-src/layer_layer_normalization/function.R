layer_layer_normalization <-
function (object, axis = -1L, epsilon = 0.001, center = TRUE, 
    scale = TRUE, rms_scaling = FALSE, beta_initializer = "zeros", 
    gamma_initializer = "ones", beta_regularizer = NULL, gamma_regularizer = NULL, 
    beta_constraint = NULL, gamma_constraint = NULL, ...) 
{
    args <- capture_args2(list(axis = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$LayerNormalization, object, args)
}
