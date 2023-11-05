metric_auc <-
function (..., num_thresholds = 200L, curve = "ROC", summation_method = "interpolation", 
    name = NULL, dtype = NULL, thresholds = NULL, multi_label = FALSE, 
    num_labels = NULL, label_weights = NULL, from_logits = FALSE) 
{
    args <- capture_args2(list(num_thresholds = as_integer))
    do.call(keras$metrics$AUC, args)
}
