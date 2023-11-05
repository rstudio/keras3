optimizer_adadelta <-
function (learning_rate = 0.001, rho = 0.95, epsilon = 1e-07, 
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL, 
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL, 
    name = "adadelta", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adadelta, args)
}
