callback_tensorboard <-
function (log_dir = "logs", histogram_freq = 0L, write_graph = TRUE, 
    write_images = FALSE, write_steps_per_second = FALSE, update_freq = "epoch", 
    profile_batch = 0L, embeddings_freq = 0L, embeddings_metadata = NULL) 
{
    args <- capture_args2(list(histogram_freq = as_integer, update_freq = as_integer, 
        profile_batch = as_integer, embeddings_freq = as_integer))
    do.call(keras$callbacks$TensorBoard, args)
}
