metric_specificity_at_sensitivity <-
function (..., sensitivity, num_thresholds = 200L, class_id = NULL, 
    name = NULL, dtype = NULL) 
{
    args <- capture_args2(list(num_thresholds = as_integer, class_id = as_integer))
    do.call(keras$metrics$SpecificityAtSensitivity, args)
}
