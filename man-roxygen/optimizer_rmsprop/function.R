optimizer_rmsprop <-
function (learning_rate = 0.001, rho = 0.9, momentum = 0, epsilon = 1e-07, 
    centered = FALSE, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, 
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99, 
    ema_overwrite_frequency = 100L, name = "rmsprop", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$RMSprop, args)
}
