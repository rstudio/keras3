layer_activity_regularization <-
function (object, l1 = 0, l2 = 0, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ActivityRegularization, object, 
        args)
}
