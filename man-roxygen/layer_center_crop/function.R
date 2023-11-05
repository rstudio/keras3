layer_center_crop <-
function (object, height, width, data_format = NULL, ...) 
{
    args <- capture_args2(list(height = as_integer, width = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$CenterCrop, object, args)
}
