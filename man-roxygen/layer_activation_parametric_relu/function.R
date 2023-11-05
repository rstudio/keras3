layer_activation_parametric_relu <-
function (object, alpha_initializer = "Zeros", alpha_regularizer = NULL, 
    alpha_constraint = NULL, shared_axes = NULL, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        shared_axes = function (x) 
        lapply(x, as_integer)), ignore = "object")
    create_layer(keras$layers$PReLU, object, args)
}
