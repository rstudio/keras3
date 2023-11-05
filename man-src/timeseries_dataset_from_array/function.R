timeseries_dataset_from_array <-
function (data, targets, sequence_length, sequence_stride = 1L, 
    sampling_rate = 1L, batch_size = 128L, shuffle = FALSE, seed = NULL, 
    start_index = NULL, end_index = NULL) 
{
    args <- capture_args2(list(sequence_stride = as_integer, 
        sampling_rate = as_integer, batch_size = as_integer, 
        seed = as_integer, start_index = as_integer, end_index = as_integer))
    do.call(keras$utils$timeseries_dataset_from_array, args)
}
