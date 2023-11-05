optimizer_ftrl <-
function (learning_rate = 0.001, learning_rate_power = -0.5, 
    initial_accumulator_value = 0.1, l1_regularization_strength = 0, 
    l2_regularization_strength = 0, l2_shrinkage_regularization_strength = 0, 
    beta = 0, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, 
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99, 
    ema_overwrite_frequency = NULL, name = "ftrl", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Ftrl, args)
}
