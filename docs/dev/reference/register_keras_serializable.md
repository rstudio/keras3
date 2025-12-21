# Registers a custom object with the Keras serialization framework.

This function registers a custom class or function with the Keras custom
object registry, so that it can be serialized and deserialized without
needing an entry in the user-provided `custom_objects` argument. It also
injects a function that Keras will call to get the object's serializable
string key.

Note that to be serialized and deserialized, classes must implement the
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
method. Functions do not have this requirement.

The object will be registered under the key `'package>name'` where
`name`, defaults to the object name if not passed.

## Usage

``` r
register_keras_serializable(object, name = NULL, package = NULL)
```

## Arguments

- object:

  A keras object.

- name:

  The name to serialize this class under in this package.

- package:

  The package that this class belongs to. This is used for the `key`
  (which is `"package>name"`) to identify the class. Defaults to the
  current package name, or `"Custom"` outside of a package.

## Value

The registered `object` (and converted) is returned. This returned
object is what you should must use when building and serializing the
model.

## Examples

    # Note that `'my_package'` is used as the `package` argument here, and since
    # the `name` argument is not provided, `'MyDense'` is used as the `name`.
    layer_my_dense <- Layer("MyDense")
    layer_my_dense <-
      register_keras_serializable(layer_my_dense, package = "my_package")

    MyDense <- environment(layer_my_dense)$`__class__` # the python class obj
    stopifnot(exprs = {
      get_registered_object('my_package>MyDense') == MyDense
      get_registered_name(MyDense) == 'my_package>MyDense'
    })

## See also

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/dev/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/dev/reference/load_model_weights.md)  
[`save_model()`](https://keras3.posit.co/dev/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/dev/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/dev/reference/deserialize_keras_object.md)  
[`get_custom_objects()`](https://keras3.posit.co/dev/reference/get_custom_objects.md)  
[`get_registered_name()`](https://keras3.posit.co/dev/reference/get_registered_name.md)  
[`get_registered_object()`](https://keras3.posit.co/dev/reference/get_registered_object.md)  
[`serialize_keras_object()`](https://keras3.posit.co/dev/reference/serialize_keras_object.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  
