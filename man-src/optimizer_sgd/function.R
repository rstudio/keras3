optimizer_sgd <-
function (learning_rate = 0.01, momentum = 0, nesterov = FALSE, 
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL, 
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL, 
    name = "SGD", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$SGD, args)
}
