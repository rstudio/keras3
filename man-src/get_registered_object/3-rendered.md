Returns the class associated with `name` if it is registered with Keras.

@description
This function is part of the Keras serialization and deserialization
framework. It maps strings to the objects associated with them for
serialization/deserialization.

# Examples

```r
from_config <- function(cls, config, custom_objects = NULL) {
  if ('my_custom_object_name' %in% names(config)) {
    config$hidden_cls <- get_registered_object(
      config$my_custom_object_name,
      custom_objects = custom_objects)
  }
}
```

@returns
An instantiable class associated with `name`, or `NULL` if no such class
exists.

@param name
The name to look up.

@param custom_objects
A named list of custom objects to look the name up in.
Generally, custom_objects is provided by the user.

@param module_objects
A named list of custom objects to look the name up in.
Generally, module_objects is provided by midlevel library
implementers.

@export
@family saving
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_registered_object>
