audio_dataset_from_directory <-
function (directory, labels = "inferred", label_mode = "int", 
    class_names = NULL, batch_size = 32L, sampling_rate = NULL, 
    output_sequence_length = NULL, ragged = FALSE, shuffle = TRUE, 
    seed = NULL, validation_split = NULL, subset = NULL, follow_links = FALSE) 
{
    args <- capture_args2(list(labels = as_integer, label_mode = as_integer, 
        batch_size = as_integer, seed = as_integer))
    do.call(keras$utils$audio_dataset_from_directory, args)
}
