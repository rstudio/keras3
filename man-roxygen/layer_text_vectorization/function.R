layer_text_vectorization <-
function (object, max_tokens = NULL, standardize = "lower_and_strip_punctuation", 
    split = "whitespace", ngrams = NULL, output_mode = "int", 
    output_sequence_length = NULL, pad_to_max_tokens = FALSE, 
    vocabulary = NULL, idf_weights = NULL, sparse = FALSE, ragged = FALSE, 
    encoding = "utf-8", name = NULL, ...) 
{
    args <- capture_args2(list(max_tokens = as_integer, ngrams = function (x) 
    if (length(x) > 1) 
        as_integer_tuple(x)
    else as_integer(x), output_mode = as_integer, output_sequence_length = as_integer, 
        ragged = as_integer, input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$TextVectorization, object, args)
}
