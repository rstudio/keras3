layer_additive_attention <-
function (object, use_scale = TRUE, dropout = 0, ...) 
{
    args <- capture_args2(list(input_shape = normalize_shape, 
        batch_size = as_integer, batch_input_shape = normalize_shape), 
        ignore = "object")
    create_layer(keras$layers$AdditiveAttention, object, args)
}
