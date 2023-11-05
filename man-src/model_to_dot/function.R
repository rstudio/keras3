model_to_dot <-
function (model, show_shapes = FALSE, show_dtype = FALSE, show_layer_names = TRUE, 
    rankdir = "TB", expand_nested = FALSE, dpi = 200L, subgraph = FALSE, 
    show_layer_activations = FALSE, show_trainable = FALSE, ...) 
{
    args <- capture_args2(list(dpi = as_integer))
    do.call(keras$utils$model_to_dot, args)
}
