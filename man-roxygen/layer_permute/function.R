layer_permute <-
function (object, dims, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        dims = function (x) 
        tuple(lapply(x, as_integer))), ignore = "object")
    create_layer(keras$layers$Permute, object, args)
}
