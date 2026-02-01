# Returns the class associated with `name` if it is registered with Keras.

This function is part of the Keras serialization and deserialization
framework. It maps strings to the objects associated with them for
serialization/deserialization.

## Usage

``` r
get_registered_object(name, custom_objects = NULL, module_objects = NULL)
```

## Arguments

- name:

  The name to look up.

- custom_objects:

  A named list of custom objects to look the name up in. Generally,
  custom_objects is provided by the user.

- module_objects:

  A named list of custom objects to look the name up in. Generally,
  `module_objects` is provided by midlevel library implementers.

## Value

An instantiable class associated with `name`, or `NULL` if no such class
exists.

## Examples

    from_config <- function(cls, config, custom_objects = NULL) {
      if ('my_custom_object_name' \%in\% names(config)) {
        config$hidden_cls <- get_registered_object(
          config$my_custom_object_name,
          custom_objects = custom_objects)
      }
    }

## See also

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md)  
[`get_custom_objects()`](https://keras3.posit.co/reference/get_custom_objects.md)  
[`get_registered_name()`](https://keras3.posit.co/reference/get_registered_name.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`serialize_keras_object()`](https://keras3.posit.co/reference/serialize_keras_object.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
