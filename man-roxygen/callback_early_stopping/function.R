callback_early_stopping <-
function (monitor = "val_loss", min_delta = 0L, patience = 0L, 
    verbose = 0L, mode = "auto", baseline = NULL, restore_best_weights = FALSE, 
    start_from_epoch = 0L) 
{
    args <- capture_args2(list(min_delta = as_integer, patience = as_integer, 
        verbose = as_integer, start_from_epoch = as_integer))
    do.call(keras$callbacks$EarlyStopping, args)
}
