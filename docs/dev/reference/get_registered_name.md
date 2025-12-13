# Returns the name registered to an object within the Keras framework.

This function is part of the Keras serialization and deserialization
framework. It maps objects to the string names associated with those
objects for serialization/deserialization.

## Usage

``` r
get_registered_name(obj)
```

## Arguments

- obj:

  The object to look up.

## Value

The name associated with the object, or the default name if the object
is not registered.

## See also

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/dev/reference/deserialize_keras_object.md)  
[`get_custom_objects()`](https://keras3.posit.co/dev/reference/get_custom_objects.md)  
[`get_registered_object()`](https://keras3.posit.co/dev/reference/get_registered_object.md)  
[`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)  
[`serialize_keras_object()`](https://keras3.posit.co/dev/reference/serialize_keras_object.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  
