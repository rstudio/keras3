k_segment_max <-
function (data, segment_ids, num_segments = NULL, sorted = FALSE) 
{
    args <- capture_args2(list(num_segments = as_integer))
    do.call(keras$ops$segment_max, args)
}
