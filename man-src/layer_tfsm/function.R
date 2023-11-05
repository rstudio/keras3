layer_tfsm <-
function (object, filepath, call_endpoint = "serve", call_training_endpoint = NULL, 
    trainable = TRUE, name = NULL, dtype = NULL) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$TFSMLayer, object, args)
}
