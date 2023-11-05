k_image_pad_images <-
function (images, top_padding = NULL, left_padding = NULL, target_height = NULL, 
    target_width = NULL, bottom_padding = NULL, right_padding = NULL) 
keras$ops$image$pad_images(images, top_padding, left_padding, 
    target_height, target_width, bottom_padding, right_padding)
