optimizer_adafactor <-
function (learning_rate = 0.001, beta_2_decay = -0.8, epsilon_1 = 1e-30, 
    epsilon_2 = 0.001, clip_threshold = 1, relative_step = TRUE, 
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL, 
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL, 
    name = "adafactor", ..., loss_scale_factor = NULL) 
{
    args <- capture_args2(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adafactor, args)
}
