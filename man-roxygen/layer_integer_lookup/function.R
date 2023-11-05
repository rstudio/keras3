layer_integer_lookup <-
function (object, max_tokens = NULL, num_oov_indices = 1L, mask_token = NULL, 
    oov_token = -1L, vocabulary = NULL, vocabulary_dtype = "int64", 
    idf_weights = NULL, invert = FALSE, output_mode = "int", 
    sparse = FALSE, pad_to_max_tokens = FALSE, name = NULL, ...) 
{
    args <- capture_args2(list(num_oov_indices = as_integer, 
        mask_token = as_integer, oov_token = as_integer, vocabulary = as_integer, 
        invert = as_integer, output_mode = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$IntegerLookup, object, args)
}
