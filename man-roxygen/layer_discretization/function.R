layer_discretization <-
function (object, bin_boundaries = NULL, num_bins = NULL, epsilon = 0.01, 
    output_mode = "int", sparse = FALSE, dtype = NULL, name = NULL) 
{
    args <- capture_args2(list(num_bins = as_integer, output_mode = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Discretization, object, args)
}
