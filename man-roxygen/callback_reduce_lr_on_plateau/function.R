callback_reduce_lr_on_plateau <-
function (monitor = "val_loss", factor = 0.1, patience = 10L, 
    verbose = 0L, mode = "auto", min_delta = 1e-04, cooldown = 0L, 
    min_lr = 0L, ...) 
{
    args <- capture_args2(list(patience = as_integer, verbose = as_integer, 
        cooldown = as_integer, min_lr = as_integer))
    do.call(keras$callbacks$ReduceLROnPlateau, args)
}
