layer_zero_padding_2d <-
function (object, padding = list(1L, 1L), data_format = NULL, 
    ...) 
{
    args <- capture_args2(list(padding = function (x) 
    normalize_padding(x, 2L), input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ZeroPadding2D, object, args)
}
