layer_category_encoding <-
function (object, num_tokens = NULL, output_mode = "multi_hot", 
    ...) 
{
    args <- capture_args2(list(output_mode = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape, 
        num_tokens = as_integer), ignore = "object")
    create_layer(keras$layers$CategoryEncoding, object, args)
}
