image_dataset_from_directory <-
function (directory, labels = "inferred", label_mode = "int", 
    class_names = NULL, color_mode = "rgb", batch_size = 32L, 
    image_size = list(256L, 256L), shuffle = TRUE, seed = NULL, 
    validation_split = NULL, subset = NULL, interpolation = "bilinear", 
    follow_links = FALSE, crop_to_aspect_ratio = FALSE, data_format = NULL) 
{
    args <- capture_args2(list(labels = as_integer, label_mode = as_integer, 
        batch_size = as_integer, seed = as_integer))
    do.call(keras$utils$image_dataset_from_directory, args)
}
