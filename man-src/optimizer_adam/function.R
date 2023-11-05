optimizer_adam <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, 
    epsilon = 1e-07, amsgrad = FALSE, weight_decay = NULL, clipnorm = NULL, 
    clipvalue = NULL, global_clipnorm = NULL, use_ema = FALSE, 
    ema_momentum = 0.99, ema_overwrite_frequency = NULL, name = "adam", 
    ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adam, args)
}
