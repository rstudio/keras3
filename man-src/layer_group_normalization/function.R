layer_group_normalization <-
function (object, groups = 32L, axis = -1L, epsilon = 0.001, 
    center = TRUE, scale = TRUE, beta_initializer = "zeros", 
    gamma_initializer = "ones", beta_regularizer = NULL, gamma_regularizer = NULL, 
    beta_constraint = NULL, gamma_constraint = NULL, ...) 
{
    args <- capture_args2(list(groups = as_integer, axis = as_axis, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$GroupNormalization, object, args)
}
