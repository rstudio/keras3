optimizer_loss_scale <-
function (inner_optimizer, initial_scale = 32768, dynamic_growth_steps = 2000L, 
    ..., name = NULL, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, 
    global_clipnorm = NULL, use_ema = NULL, ema_momentum = NULL, 
    ema_overwrite_frequency = NULL, loss_scale_factor = NULL) 
{
    args <- capture_args2(list(dynamic_growth_steps = as_integer, 
        ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$LossScaleOptimizer, args)
}
