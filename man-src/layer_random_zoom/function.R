layer_random_zoom <-
function (object, height_factor, width_factor = NULL, fill_mode = "reflect", 
    interpolation = "bilinear", seed = NULL, fill_value = 0, 
    data_format = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RandomZoom, object, args)
}
