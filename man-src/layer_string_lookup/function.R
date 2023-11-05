layer_string_lookup <-
function (object, max_tokens = NULL, num_oov_indices = 1L, mask_token = NULL, 
    oov_token = "[UNK]", vocabulary = NULL, idf_weights = NULL, 
    invert = FALSE, output_mode = "int", pad_to_max_tokens = FALSE, 
    sparse = FALSE, encoding = "utf-8", name = NULL, ..., vocabulary_dtype = NULL) 
{
    args <- capture_args2(list(num_oov_indices = as_integer, 
        mask_token = as_integer, vocabulary = as_integer, invert = as_integer, 
        output_mode = as_integer, input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$StringLookup, object, args)
}
