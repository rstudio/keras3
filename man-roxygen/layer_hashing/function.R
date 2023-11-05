layer_hashing <-
function (object, num_bins, mask_value = NULL, salt = NULL, output_mode = "int", 
    sparse = FALSE, ...) 
{
    args <- capture_args2(list(salt = as_integer, output_mode = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape, num_bins = as_integer), 
        ignore = "object")
    create_layer(keras$layers$Hashing, object, args)
}
