layer_embedding <-
function (object, input_dim, output_dim, embeddings_initializer = "uniform", 
    embeddings_regularizer = NULL, embeddings_constraint = NULL, 
    mask_zero = FALSE, ...) 
{
    args <- capture_args2(list(input_dim = as_integer, output_dim = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape, input_length = as_integer), 
        ignore = "object")
    create_layer(keras$layers$Embedding, object, args)
}
