layer_spectral_normalization <-
function (object, layer, power_iterations = 1L, ...) 
{
    args <- capture_args2(list(power_iterations = as_integer, 
        input_shape = normalize_shape, batch_size = as_integer, 
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SpectralNormalization, object, 
        args)
}
