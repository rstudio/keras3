k_image_resize <-
function (image, size, interpolation = "bilinear", antialias = FALSE, 
    data_format = "channels_last") 
keras$ops$image$resize(image, size, interpolation, antialias, 
    data_format)
