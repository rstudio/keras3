callback_lambda <-
function (on_epoch_begin = NULL, on_epoch_end = NULL, on_train_begin = NULL, 
    on_train_end = NULL, on_train_batch_begin = NULL, on_train_batch_end = NULL, 
    ...) 
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$LambdaCallback, args)
}
