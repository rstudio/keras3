layer_zero_padding_3d <-
function (object, padding = list(list(1L, 1L), list(1L, 1L), 
    list(1L, 1L)), data_format = NULL, ...) 
{
    args <- capture_args2(list(padding = function (x) 
    normalize_padding(x, 3L), input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$ZeroPadding3D, object, args)
}
