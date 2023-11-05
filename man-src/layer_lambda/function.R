layer_lambda <-
function (object, f, output_shape = NULL, mask = NULL, arguments = NULL, 
    ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        output_shape = normalize_shape), ignore = "object")
    names(args)[match("f", names(args))] <- "function"
    create_layer(keras$layers$Lambda, object, args)
}
