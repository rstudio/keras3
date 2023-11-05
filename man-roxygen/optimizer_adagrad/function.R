optimizer_adagrad <-
function (learning_rate = 0.001, initial_accumulator_value = 0.1, 
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, 
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99, 
    ema_overwrite_frequency = NULL, name = "adagrad", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adagrad, args)
}
