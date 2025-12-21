# Retrieve the full config by serializing the Keras object.

`serialize_keras_object()` serializes a Keras object to a named list
that represents the object, and is a reciprocal function of
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md).
See
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md)
for more information about the full config format.

## Usage

``` r
serialize_keras_object(obj)
```

## Arguments

- obj:

  the Keras object to serialize.

## Value

A named list that represents the object config. The config is expected
to contain simple types only, and can be saved as json. The object can
be deserialized from the config via
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md).

## See also

- <https://keras.io/api/models/model_saving_apis/serialization_utils#serializekerasobject-function>

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md)  
[`get_custom_objects()`](https://keras3.posit.co/reference/get_custom_objects.md)  
[`get_registered_name()`](https://keras3.posit.co/reference/get_registered_name.md)  
[`get_registered_object()`](https://keras3.posit.co/reference/get_registered_object.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
