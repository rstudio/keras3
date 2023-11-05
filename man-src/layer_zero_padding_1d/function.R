layer_zero_padding_1d <-
function (object, padding = 1L, ...) 
{
    args <- capture_args2(list(padding = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ZeroPadding1D, object, args)
}
