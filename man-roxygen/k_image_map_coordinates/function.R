k_image_map_coordinates <-
function (input, coordinates, order, fill_mode = "constant", 
    fill_value = 0L) 
{
    args <- capture_args2(list(fill_value = as_integer))
    do.call(keras$ops$image$map_coordinates, args)
}
