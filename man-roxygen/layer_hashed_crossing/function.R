layer_hashed_crossing <-
function (object, num_bins, output_mode = "int", sparse = FALSE, 
    name = NULL, dtype = NULL, ...) 
{
    args <- capture_args2(list(output_mode = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$HashedCrossing, object, args)
}
