callback_model_checkpoint <-
function (filepath, monitor = "val_loss", verbose = 0L, save_best_only = FALSE, 
    save_weights_only = FALSE, mode = "auto", save_freq = "epoch", 
    initial_value_threshold = NULL) 
{
    args <- capture_args2(list(verbose = as_integer, save_freq = as_integer))
    do.call(keras$callbacks$ModelCheckpoint, args)
}
