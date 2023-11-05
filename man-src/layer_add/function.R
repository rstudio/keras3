layer_add <-
function (inputs, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs)) 
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs)) 
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- do.call(keras$layers$Add, args)
    if (length(inputs)) 
        layer(inputs)
    else layer
}
