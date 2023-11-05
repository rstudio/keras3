layer_rnn <-
function (object, cell, return_sequences = FALSE, return_state = FALSE, 
    go_backwards = FALSE, stateful = FALSE, unroll = FALSE, zero_output_for_mask = FALSE, 
    ...) 
{
    args <- capture_args2(list(cell = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$RNN, object, args)
}
