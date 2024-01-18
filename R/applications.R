#' Instantiates the ConvNeXtBase architecture.
#'
#' @description
#'
#' # References
#' - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
#' (CVPR 2022)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' The `base`, `large`, and `xlarge` models were first pre-trained on the
#' ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
#' pre-trained parameters of the models were assembled from the
#' [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
#' sense of how these parameters were converted to Keras compatible parameters,
#' please refer to
#' [this repository](https://github.com/sayakpaul/keras-convnext-conversion).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ConvNeXt, preprocessing is included in the model using a `Normalization`
#' layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
#' pixels with values in the `[0-255]` range.
#'
#' When calling the `summary()` method after instantiating a ConvNeXt model,
#' prefer setting the `expand_nested` argument `summary()` to `TRUE` to better
#' investigate the instantiated model.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
#' file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#' the 4D tensor output of the last convolutional layer.
#' - `avg` means that global average pooling
#' will be applied to the output of the
#' last convolutional layer, and thus
#' the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#' be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @param model_name
#' String, name for the model.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/convnext#convnextbase-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtBase>
#' @tether keras.applications.ConvNeXtBase
application_convnext_base <-
function (model_name = "convnext_base", include_top = TRUE, include_preprocessing = TRUE,
    weights = "imagenet", input_tensor = NULL, input_shape = NULL,
    pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ConvNeXtBase, args)
    set_preprocessing_attributes(model, keras$applications$convnext)
}


#' Instantiates the ConvNeXtLarge architecture.
#'
#' @description
#'
#' # References
#' - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
#' (CVPR 2022)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' The `base`, `large`, and `xlarge` models were first pre-trained on the
#' ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
#' pre-trained parameters of the models were assembled from the
#' [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
#' sense of how these parameters were converted to Keras compatible parameters,
#' please refer to
#' [this repository](https://github.com/sayakpaul/keras-convnext-conversion).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ConvNeXt, preprocessing is included in the model using a `Normalization`
#' layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
#' pixels with values in the `[0-255]` range.
#'
#' When calling the `summary()` method after instantiating a ConvNeXt model,
#' prefer setting the `expand_nested` argument `summary()` to `TRUE` to better
#' investigate the instantiated model.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
#' file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#' the 4D tensor output of the last convolutional layer.
#' - `avg` means that global average pooling
#' will be applied to the output of the
#' last convolutional layer, and thus
#' the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#' be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @param model_name
#' String, name for the model.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/convnext#convnextlarge-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtLarge>
#' @tether keras.applications.ConvNeXtLarge
application_convnext_large <-
function (model_name = "convnext_large", include_top = TRUE,
    include_preprocessing = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ConvNeXtLarge, args)
    set_preprocessing_attributes(model, keras$applications$convnext)
}


#' Instantiates the ConvNeXtSmall architecture.
#'
#' @description
#'
#' # References
#' - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
#' (CVPR 2022)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' The `base`, `large`, and `xlarge` models were first pre-trained on the
#' ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
#' pre-trained parameters of the models were assembled from the
#' [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
#' sense of how these parameters were converted to Keras compatible parameters,
#' please refer to
#' [this repository](https://github.com/sayakpaul/keras-convnext-conversion).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ConvNeXt, preprocessing is included in the model using a `Normalization`
#' layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
#' pixels with values in the `[0-255]` range.
#'
#' When calling the `summary()` method after instantiating a ConvNeXt model,
#' prefer setting the `expand_nested` argument `summary()` to `TRUE` to better
#' investigate the instantiated model.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
#' file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#' the 4D tensor output of the last convolutional layer.
#' - `avg` means that global average pooling
#' will be applied to the output of the
#' last convolutional layer, and thus
#' the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#' be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @param model_name
#' String, name for the model.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/convnext#convnextsmall-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtSmall>
#' @tether keras.applications.ConvNeXtSmall
application_convnext_small <-
function (model_name = "convnext_small", include_top = TRUE,
    include_preprocessing = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ConvNeXtSmall, args)
    set_preprocessing_attributes(model, keras$applications$convnext)
}


#' Instantiates the ConvNeXtTiny architecture.
#'
#' @description
#'
#' # References
#' - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
#' (CVPR 2022)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' The `base`, `large`, and `xlarge` models were first pre-trained on the
#' ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
#' pre-trained parameters of the models were assembled from the
#' [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
#' sense of how these parameters were converted to Keras compatible parameters,
#' please refer to
#' [this repository](https://github.com/sayakpaul/keras-convnext-conversion).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ConvNeXt, preprocessing is included in the model using a `Normalization`
#' layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
#' pixels with values in the `[0-255]` range.
#'
#' When calling the `summary()` method after instantiating a ConvNeXt model,
#' prefer setting the `expand_nested` argument `summary()` to `TRUE` to better
#' investigate the instantiated model.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
#' file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#' the 4D tensor output of the last convolutional layer.
#' - `avg` means that global average pooling
#' will be applied to the output of the
#' last convolutional layer, and thus
#' the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#' be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @param model_name
#' String, name for the model.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/convnext#convnexttiny-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtTiny>
#' @tether keras.applications.ConvNeXtTiny
application_convnext_tiny <-
function (model_name = "convnext_tiny", include_top = TRUE, include_preprocessing = TRUE,
    weights = "imagenet", input_tensor = NULL, input_shape = NULL,
    pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ConvNeXtTiny, args)
    set_preprocessing_attributes(model, keras$applications$convnext)
}


#' Instantiates the ConvNeXtXLarge architecture.
#'
#' @description
#'
#' # References
#' - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
#' (CVPR 2022)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' The `base`, `large`, and `xlarge` models were first pre-trained on the
#' ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
#' pre-trained parameters of the models were assembled from the
#' [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
#' sense of how these parameters were converted to Keras compatible parameters,
#' please refer to
#' [this repository](https://github.com/sayakpaul/keras-convnext-conversion).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ConvNeXt, preprocessing is included in the model using a `Normalization`
#' layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
#' pixels with values in the `[0-255]` range.
#'
#' When calling the `summary()` method after instantiating a ConvNeXt model,
#' prefer setting the `expand_nested` argument `summary()` to `TRUE` to better
#' investigate the instantiated model.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights
#' file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#' the 4D tensor output of the last convolutional layer.
#' - `avg` means that global average pooling
#' will be applied to the output of the
#' last convolutional layer, and thus
#' the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#' be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @param model_name
#' String, name for the model.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/convnext#convnextxlarge-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtXLarge>
#' @tether keras.applications.ConvNeXtXLarge
application_convnext_xlarge <-
function (model_name = "convnext_xlarge", include_top = TRUE,
    include_preprocessing = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ConvNeXtXLarge, args)
    set_preprocessing_attributes(model, keras$applications$convnext)
}


#' Instantiates the Densenet121 architecture.
#'
#' @description
#'
#' # Reference
#' - [Densely Connected Convolutional Networks](
#'     https://arxiv.org/abs/1608.06993) (CVPR 2017)
#'
#' Optionally loads weights pre-trained on ImageNet.
#' Note that the data format convention used by the model is
#' the one specified in your Keras config at `~/.keras/keras.json`.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For DenseNet, call [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#'
#' @returns
#' A Keras model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)` (with `'channels_last'` data format)
#' or `(3, 224, 224)` (with `'channels_first'` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable.
#' The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits
#' of the "top" layer. When loading pretrained weights,
#' `classifier_activation` can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/densenet#densenet121-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121>
#' @tether keras.applications.DenseNet121
application_densenet121 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$DenseNet121, args)
    set_preprocessing_attributes(model, keras$applications$densenet)
}


#' Instantiates the Densenet169 architecture.
#'
#' @description
#'
#' # Reference
#' - [Densely Connected Convolutional Networks](
#'     https://arxiv.org/abs/1608.06993) (CVPR 2017)
#'
#' Optionally loads weights pre-trained on ImageNet.
#' Note that the data format convention used by the model is
#' the one specified in your Keras config at `~/.keras/keras.json`.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For DenseNet, call [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#'
#' @returns
#' A Keras model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)` (with `'channels_last'` data format)
#' or `(3, 224, 224)` (with `'channels_first'` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable.
#' The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits
#' of the "top" layer. When loading pretrained weights,
#' `classifier_activation` can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/densenet#densenet169-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169>
#' @tether keras.applications.DenseNet169
application_densenet169 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$DenseNet169, args)
    set_preprocessing_attributes(model, keras$applications$densenet)
}


#' Instantiates the Densenet201 architecture.
#'
#' @description
#'
#' # Reference
#' - [Densely Connected Convolutional Networks](
#'     https://arxiv.org/abs/1608.06993) (CVPR 2017)
#'
#' Optionally loads weights pre-trained on ImageNet.
#' Note that the data format convention used by the model is
#' the one specified in your Keras config at `~/.keras/keras.json`.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For DenseNet, call [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#'
#' @returns
#' A Keras model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)` (with `'channels_last'` data format)
#' or `(3, 224, 224)` (with `'channels_first'` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable.
#' The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits
#' of the "top" layer. When loading pretrained weights,
#' `classifier_activation` can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/densenet#densenet201-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201>
#' @tether keras.applications.DenseNet201
application_densenet201 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$DenseNet201, args)
    set_preprocessing_attributes(model, keras$applications$densenet)
}


#' Instantiates the EfficientNetB0 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb0-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0>
#' @tether keras.applications.EfficientNetB0
application_efficientnet_b0 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB0, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB1 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb1-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1>
#' @tether keras.applications.EfficientNetB1
application_efficientnet_b1 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB1, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB2 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2>
#' @tether keras.applications.EfficientNetB2
application_efficientnet_b2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB2, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB3 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb3-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3>
#' @tether keras.applications.EfficientNetB3
application_efficientnet_b3 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB3, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB4 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb4-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4>
#' @tether keras.applications.EfficientNetB4
application_efficientnet_b4 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB4, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB5 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb5-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB5>
#' @tether keras.applications.EfficientNetB5
application_efficientnet_b5 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB5, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB6 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb6-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB6>
#' @tether keras.applications.EfficientNetB6
application_efficientnet_b6 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB6, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetB7 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
#'     https://arxiv.org/abs/1905.11946) (ICML 2019)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNet, input preprocessing is included as part of the model
#' (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. EfficientNet models expect their inputs to be float
#' tensors of pixels with values in the `[0-255]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is TRUE, and
#' if no `weights` argument is specified. 1000 is how many
#' ImageNet classes there are. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `'softmax'`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet#efficientnetb7-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7>
#' @tether keras.applications.EfficientNetB7
application_efficientnet_b7 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    ...)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetB7, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet)
}


#' Instantiates the EfficientNetV2B0 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to NULL.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2b0-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B0>
#' @tether keras.applications.EfficientNetV2B0
application_efficientnet_v2b0 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2B0, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2B1 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to NULL.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2b1-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B1>
#' @tether keras.applications.EfficientNetV2B1
application_efficientnet_v2b1 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2B1, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2B2 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to NULL.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2b2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B2>
#' @tether keras.applications.EfficientNetV2B2
application_efficientnet_v2b2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2B2, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2B3 architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2b3-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B3>
#' @tether keras.applications.EfficientNetV2B3
application_efficientnet_v2b3 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2B3, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2L architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2l-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2L>
#' @tether keras.applications.EfficientNetV2L
application_efficientnet_v2l <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2L, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2M architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2m-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2M>
#' @tether keras.applications.EfficientNetV2M
application_efficientnet_v2m <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2M, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the EfficientNetV2S architecture.
#'
#' @description
#'
#' # Reference
#' - [EfficientNetV2: Smaller Models and Faster Training](
#'     https://arxiv.org/abs/2104.00298) (ICML 2021)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For EfficientNetV2, by default input preprocessing is included as a part of
#' the model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, EfficientNetV2 models expect their
#' inputs to be float tensors of pixels with values in the `[0, 255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled EfficientNetV2 models expect their inputs to be
#' float tensors of pixels with values in the `[-1, 1]` range.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE`.
#' It should have exactly 3 inputs channels.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`. Defaults to `NULL`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional layer.
#' - `"avg"` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a 2D tensor.
#' - `"max"` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000 (number of
#' ImageNet classes).
#'
#' @param classifier_activation
#' A string or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' Defaults to `"softmax"`.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing layer at the bottom of the network.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/efficientnet_v2#efficientnetv2s-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2S>
#' @tether keras.applications.EfficientNetV2S
application_efficientnet_v2s <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$EfficientNetV2S, args)
    set_preprocessing_attributes(model, keras$applications$efficientnet_v2)
}


#' Instantiates the Inception-ResNet v2 architecture.
#'
#' @description
#'
#' # Reference
#' - [Inception-v4, Inception-ResNet and the Impact of
#'    Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
#'   (AAAI 2017)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of
#' input preprocessing. For `InceptionResNetV2`, call
#' [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#' [`application_preprocess_inputs()`]
#' will scale input pixels between -1 and 1.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(299, 299, 3)`
#' (with `'channels_last'` data format)
#' or `(3, 299, 299)` (with `'channels_first'` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 75.
#' E.g. `(150, 150, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the last convolutional block.
#' - `'avg'` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `'max'` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`,
#' and if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable.
#' The activation function to use on the "top" layer.
#' Ignored unless `include_top=TRUE`.
#' Set `classifier_activation=NULL` to return the logits
#' of the "top" layer. When loading pretrained weights,
#' `classifier_activation` can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/inceptionresnetv2#inceptionresnetv2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2>
#' @tether keras.applications.InceptionResNetV2
application_inception_resnet_v2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$InceptionResNetV2, args)
    set_preprocessing_attributes(model, keras$applications$inception_resnet_v2)
}


#' Instantiates the Inception v3 architecture.
#'
#' @description
#'
#' # Reference
#' - [Rethinking the Inception Architecture for Computer Vision](
#'     https://arxiv.org/abs/1512.00567) (CVPR 2016)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For `InceptionV3`, call
#' [`application_preprocess_inputs()`] on your inputs
#' before passing them to the model.
#' [`application_preprocess_inputs()`] will scale input pixels between `-1` and `1`.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top, as the last layer of the network.
#' Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization),
#' `imagenet` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#' Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model. `input_tensor` is useful for
#' sharing inputs between multiple different networks.
#' Defaults to `NULL`.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(299, 299, 3)` (with `channels_last` data format)
#' or `(3, 299, 299)` (with `channels_first` data format).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 75.
#' E.g. `(150, 150, 3)` would be one valid value.
#' `input_shape` will be ignored if the `input_tensor` is provided.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` (default) means that the output of the model will be
#'     the 4D tensor output of the last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified. Defaults to 1000.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function
#' to use on the "top" layer. Ignored unless `include_top=TRUE`.
#' Set `classifier_activation=NULL` to return the logits of the "top"
#' layer. When loading pretrained weights, `classifier_activation`
#' can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/inceptionv3#inceptionv3-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3>
#' @tether keras.applications.InceptionV3
application_inception_v3 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$InceptionV3, args)
    set_preprocessing_attributes(model, keras$applications$inception_v3)
}


#' Instantiates the MobileNet architecture.
#'
#' @description
#'
#' # Reference
#' - [MobileNets: Efficient Convolutional Neural Networks
#'    for Mobile Vision Applications](
#'     https://arxiv.org/abs/1704.04861)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For MobileNet, call [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#' [`application_preprocess_inputs()`] will scale input pixels between `-1` and `1`.
#'
#' @returns
#' A model instance.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format).
#' It should have exactly 3 inputs channels, and width and
#' height should be no smaller than 32. E.g. `(200, 200, 3)` would
#' be one valid value. Defaults to `NULL`.
#' `input_shape` will be ignored if the `input_tensor` is provided.
#'
#' @param alpha
#' Controls the width of the network. This is known as the width
#' multiplier in the MobileNet paper.
#' - If `alpha < 1.0`, proportionally decreases the number
#'     of filters in each layer.
#' - If `alpha > 1.0`, proportionally increases the number
#'     of filters in each layer.
#' - If `alpha == 1`, default number of filters from the paper
#'     are used at each layer. Defaults to `1.0`.
#'
#' @param depth_multiplier
#' Depth multiplier for depthwise convolution.
#' This is called the resolution multiplier in the MobileNet paper.
#' Defaults to `1.0`.
#'
#' @param dropout
#' Dropout rate. Defaults to `0.001`.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected layer
#' at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization), `"imagenet"`
#' (pre-training on ImageNet), or the path to the weights file
#' to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model. `input_tensor` is useful
#' for sharing inputs between multiple different networks.
#' Defaults to `NULL`.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` (default) means that the output of the model will be
#'     the 4D tensor output of the last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' Optional number of classes to classify images into,
#' only to be specified if `include_top` is `TRUE`, and if
#' no `weights` argument is specified. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function
#' to use on the "top" layer. Ignored unless `include_top=TRUE`.
#' Set `classifier_activation=NULL` to return the logits of the "top"
#' layer. When loading pretrained weights, `classifier_activation`
#' can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/mobilenet#mobilenet-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet>
#' @tether keras.applications.MobileNet
application_mobilenet <-
function (input_shape = NULL, alpha = 1, depth_multiplier = 1L,
    dropout = 0.001, include_top = TRUE, weights = "imagenet",
    input_tensor = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(depth_multiplier = as_integer,
        classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$MobileNet, args)
    set_preprocessing_attributes(model, keras$applications$mobilenet)
}


#' Instantiates the MobileNetV2 architecture.
#'
#' @description
#' MobileNetV2 is very similar to the original MobileNet,
#' except that it uses inverted residual blocks with
#' bottlenecking features. It has a drastically lower
#' parameter count than the original MobileNet.
#' MobileNets support any input size greater
#' than 32 x 32, with larger image sizes
#' offering better performance.
#'
#' # Reference
#' - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
#'     https://arxiv.org/abs/1801.04381) (CVPR 2018)
#'
#' This function returns a Keras image classification model,
#' optionally loaded with weights pre-trained on ImageNet.
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For MobileNetV2, call
#' [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#' [`application_preprocess_inputs()`] will scale input pixels between `-1` and `1`.
#'
#' @returns
#' A model instance.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format).
#' It should have exactly 3 inputs channels, and width and
#' height should be no smaller than 32. E.g. `(200, 200, 3)` would
#' be one valid value. Defaults to `NULL`.
#' `input_shape` will be ignored if the `input_tensor` is provided.
#'
#' @param alpha
#' Controls the width of the network. This is known as the width
#' multiplier in the MobileNet paper.
#' - If `alpha < 1.0`, proportionally decreases the number
#'     of filters in each layer.
#' - If `alpha > 1.0`, proportionally increases the number
#'     of filters in each layer.
#' - If `alpha == 1`, default number of filters from the paper
#'     are used at each layer. Defaults to `1.0`.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected layer
#' at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' One of `NULL` (random initialization), `"imagenet"`
#' (pre-training on ImageNet), or the path to the weights file
#' to be loaded. Defaults to `"imagenet"`.
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model. `input_tensor` is useful
#' for sharing inputs between multiple different networks.
#' Defaults to `NULL`.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` (default) means that the output of the model will be
#'     the 4D tensor output of the last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' Optional number of classes to classify images into,
#' only to be specified if `include_top` is `TRUE`, and if
#' no `weights` argument is specified. Defaults to `1000`.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function
#' to use on the "top" layer. Ignored unless `include_top=TRUE`.
#' Set `classifier_activation=NULL` to return the logits of the "top"
#' layer. When loading pretrained weights, `classifier_activation`
#' can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/mobilenet#mobilenetv2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2>
#' @tether keras.applications.MobileNetV2
application_mobilenet_v2 <-
function (input_shape = NULL, alpha = 1, include_top = TRUE,
    weights = "imagenet", input_tensor = NULL, pooling = NULL,
    classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$MobileNetV2, args)
    set_preprocessing_attributes(model, keras$applications$mobilenet_v2)
}


#' Instantiates the MobileNetV3Large architecture.
#'
#' @description
#'
#' # Reference
#' - [Searching for MobileNetV3](
#'     https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
#'
#' The following table describes the performance of MobileNets v3:
#' ------------------------------------------------------------------------
#' MACs stands for Multiply Adds
#'
#' |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
#' |---|---|---|---|---|
#' | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
#' | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
#' | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
#' | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
#' | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
#' | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For MobileNetV3, by default input preprocessing is included as a part of the
#' model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, MobileNetV3 models expect their
#' inputs to be float tensors of pixels with values in the `[0-255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled MobileNetV3 models expect their inputs to be float
#' tensors of pixels with values in the `[-1, 1]` range.
#'
#' # Call Arguments
#' - `inputs`: A floating point `numpy.array` or backend-native tensor,
#' 4D with 3 color channels, with values in the range `[0, 255]`
#' if `include_preprocessing` is `TRUE` and in the range `[-1, 1]`
#' otherwise.
#'
#' @returns
#' A model instance.
#'
#' @param input_shape
#' Optional shape tuple, to be specified if you would
#' like to use a model with an input image resolution that is not
#' `(224, 224, 3)`.
#' It should have exactly 3 inputs channels.
#' You can also omit this option if you would like
#' to infer input_shape from an input_tensor.
#' If you choose to include both input_tensor and input_shape then
#' input_shape will be used if they match, if the shapes
#' do not match then we will throw an error.
#' E.g. `(160, 160, 3)` would be one valid value.
#'
#' @param alpha
#' controls the width of the network. This is known as the
#' depth multiplier in the MobileNetV3 paper, but the name is kept for
#' consistency with MobileNetV1 in Keras.
#' - If `alpha < 1.0`, proportionally decreases the number
#'     of filters in each layer.
#' - If `alpha > 1.0`, proportionally increases the number
#'     of filters in each layer.
#' - If `alpha == 1`, default number of filters from the paper
#'     are used at each layer.
#'
#' @param minimalistic
#' In addition to large and small models this module also
#' contains so-called minimalistic models, these models have the same
#' per-layer dimensions characteristic as MobilenetV3 however, they don't
#' utilize any of the advanced blocks (squeeze-and-excite units,
#' hard-swish, and 5x5 convolutions).
#' While these models are less efficient on CPU, they
#' are much more performant on GPU/DSP.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' String, one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of
#' `layers.Input()`)
#' to use as image input for the model.
#'
#' @param pooling
#' String, optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model
#'     will be the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a
#'     2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Integer, optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param dropout_rate
#' fraction of the input units to drop on the last layer.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing
#' layer (`Rescaling`) at the bottom of the network. Defaults to `TRUE`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/mobilenet#mobilenetv3large-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large>
#' @tether keras.applications.MobileNetV3Large
application_mobilenet_v3_large <-
function (input_shape = NULL, alpha = 1, minimalistic = FALSE,
    include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    classes = 1000L, pooling = NULL, dropout_rate = 0.2, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$MobileNetV3Large, args)
    set_preprocessing_attributes(model, keras$applications$mobilenet_v3)
}


#' Instantiates the MobileNetV3Small architecture.
#'
#' @description
#'
#' # Reference
#' - [Searching for MobileNetV3](
#'     https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
#'
#' The following table describes the performance of MobileNets v3:
#' ------------------------------------------------------------------------
#' MACs stands for Multiply Adds
#'
#' |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
#' |---|---|---|---|---|
#' | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
#' | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
#' | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
#' | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
#' | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
#' | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#' https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#' https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For MobileNetV3, by default input preprocessing is included as a part of the
#' model (as a `Rescaling` layer), and thus
#' [`application_preprocess_inputs()`] is actually a
#' pass-through function. In this use case, MobileNetV3 models expect their
#' inputs to be float tensors of pixels with values in the `[0-255]` range.
#' At the same time, preprocessing as a part of the model (i.e. `Rescaling`
#' layer) can be disabled by setting `include_preprocessing` argument to `FALSE`.
#' With preprocessing disabled MobileNetV3 models expect their inputs to be float
#' tensors of pixels with values in the `[-1, 1]` range.
#'
#' # Call Arguments
#' - `inputs`: A floating point `numpy.array` or backend-native tensor,
#' 4D with 3 color channels, with values in the range `[0, 255]`
#' if `include_preprocessing` is `TRUE` and in the range `[-1, 1]`
#' otherwise.
#'
#' @returns
#' A model instance.
#'
#' @param input_shape
#' Optional shape tuple, to be specified if you would
#' like to use a model with an input image resolution that is not
#' `(224, 224, 3)`.
#' It should have exactly 3 inputs channels.
#' You can also omit this option if you would like
#' to infer input_shape from an input_tensor.
#' If you choose to include both input_tensor and input_shape then
#' input_shape will be used if they match, if the shapes
#' do not match then we will throw an error.
#' E.g. `(160, 160, 3)` would be one valid value.
#'
#' @param alpha
#' controls the width of the network. This is known as the
#' depth multiplier in the MobileNetV3 paper, but the name is kept for
#' consistency with MobileNetV1 in Keras.
#' - If `alpha < 1.0`, proportionally decreases the number
#'     of filters in each layer.
#' - If `alpha > 1.0`, proportionally increases the number
#'     of filters in each layer.
#' - If `alpha == 1`, default number of filters from the paper
#'     are used at each layer.
#'
#' @param minimalistic
#' In addition to large and small models this module also
#' contains so-called minimalistic models, these models have the same
#' per-layer dimensions characteristic as MobilenetV3 however, they don't
#' utilize any of the advanced blocks (squeeze-and-excite units,
#' hard-swish, and 5x5 convolutions).
#' While these models are less efficient on CPU, they
#' are much more performant on GPU/DSP.
#'
#' @param include_top
#' Boolean, whether to include the fully-connected
#' layer at the top of the network. Defaults to `TRUE`.
#'
#' @param weights
#' String, one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of
#' `layers.Input()`)
#' to use as image input for the model.
#'
#' @param pooling
#' String, optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model
#'     will be the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a
#'     2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Integer, optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param dropout_rate
#' fraction of the input units to drop on the last layer.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to use
#' on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @param include_preprocessing
#' Boolean, whether to include the preprocessing
#' layer (`Rescaling`) at the bottom of the network. Defaults to `TRUE`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/mobilenet#mobilenetv3small-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small>
#' @tether keras.applications.MobileNetV3Small
application_mobilenet_v3_small <-
function (input_shape = NULL, alpha = 1, minimalistic = FALSE,
    include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    classes = 1000L, pooling = NULL, dropout_rate = 0.2, classifier_activation = "softmax",
    include_preprocessing = TRUE)
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$MobileNetV3Small, args)
    set_preprocessing_attributes(model, keras$applications$mobilenet_v3)
}


#' Instantiates a NASNet model in ImageNet mode.
#'
#' @description
#'
#' # Reference
#' - [Learning Transferable Architectures for Scalable Image Recognition](
#'     https://arxiv.org/abs/1707.07012) (CVPR 2018)
#'
#' Optionally loads weights pre-trained on ImageNet.
#' Note that the data format convention used by the model is
#' the one specified in your Keras config at `~/.keras/keras.json`.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For NASNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model.
#'
#' @returns
#' A Keras model instance.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(331, 331, 3)` for NASNetLarge.
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `(224, 224, 3)` would be one valid value.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' `NULL` (random initialization) or
#' `imagenet` (ImageNet weights).  For loading `imagenet` weights,
#' `input_shape` should be (331, 331, 3)
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of
#' `layers.Input()`)
#' to use as image input for the model.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model
#'     will be the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a
#'     2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation`
#' can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/nasnet#nasnetlarge-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetLarge>
#' @tether keras.applications.NASNetLarge
application_nasnetlarge <-
function (input_shape = NULL, include_top = TRUE, weights = "imagenet",
    input_tensor = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$NASNetLarge, args)
    set_preprocessing_attributes(model, keras$applications$nasnet)
}


#' Instantiates a Mobile NASNet model in ImageNet mode.
#'
#' @description
#'
#' # Reference
#' - [Learning Transferable Architectures for Scalable Image Recognition](
#'     https://arxiv.org/abs/1707.07012) (CVPR 2018)
#'
#' Optionally loads weights pre-trained on ImageNet.
#' Note that the data format convention used by the model is
#' the one specified in your Keras config at `~/.keras/keras.json`.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For NASNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model.
#'
#' @returns
#' A Keras model instance.
#'
#' @param input_shape
#' Optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)` for NASNetMobile
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 32.
#' E.g. `(224, 224, 3)` would be one valid value.
#'
#' @param include_top
#' Whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' `NULL` (random initialization) or
#' `imagenet` (ImageNet weights). For loading `imagenet` weights,
#' `input_shape` should be (224, 224, 3)
#'
#' @param input_tensor
#' Optional Keras tensor (i.e. output of
#' `layers.Input()`)
#' to use as image input for the model.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model
#'     will be the 4D tensor output of the
#'     last convolutional layer.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional layer, and thus
#'     the output of the model will be a
#'     2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' Optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation` can
#' only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/nasnet#nasnetmobile-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile>
#' @tether keras.applications.NASNetMobile
application_nasnetmobile <-
function (input_shape = NULL, include_top = TRUE, weights = "imagenet",
    input_tensor = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer))
    model <- do.call(keras$applications$NASNetMobile, args)
    set_preprocessing_attributes(model, keras$applications$nasnet)
}


#' Instantiates the ResNet101 architecture.
#'
#' @description
#'
#' # Reference
#' - [Deep Residual Learning for Image Recognition](
#'     https://arxiv.org/abs/1512.03385) (CVPR 2015)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will convert
#' the input images from RGB to BGR, then will zero-center each color channel with
#' respect to the ImageNet dataset, without scaling.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet101-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101>
#' @tether keras.applications.ResNet101
application_resnet101 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet101, args)
    set_preprocessing_attributes(model, keras$applications$resnet)
}


#' Instantiates the ResNet152 architecture.
#'
#' @description
#'
#' # Reference
#' - [Deep Residual Learning for Image Recognition](
#'     https://arxiv.org/abs/1512.03385) (CVPR 2015)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will convert
#' the input images from RGB to BGR, then will zero-center each color channel with
#' respect to the ImageNet dataset, without scaling.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet152-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152>
#' @tether keras.applications.ResNet152
application_resnet152 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet152, args)
    set_preprocessing_attributes(model, keras$applications$resnet)
}


#' Instantiates the ResNet50 architecture.
#'
#' @description
#'
#' # Reference
#' - [Deep Residual Learning for Image Recognition](
#'     https://arxiv.org/abs/1512.03385) (CVPR 2015)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will convert
#' the input images from RGB to BGR, then will zero-center each color channel with
#' respect to the ImageNet dataset, without scaling.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet50-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50>
#' @tether keras.applications.ResNet50
application_resnet50 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet50, args)
    set_preprocessing_attributes(model, keras$applications$resnet)
}


#' Instantiates the ResNet101V2 architecture.
#'
#' @description
#'
#' # Reference
#' - [Identity Mappings in Deep Residual Networks](
#'     https://arxiv.org/abs/1603.05027) (CVPR 2016)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will
#' scale input pixels between -1 and 1.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet101v2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101V2>
#' @tether keras.applications.ResNet101V2
application_resnet101_v2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet101V2, args)
    set_preprocessing_attributes(model, keras$applications$resnet_v2)
}


#' Instantiates the ResNet152V2 architecture.
#'
#' @description
#'
#' # Reference
#' - [Identity Mappings in Deep Residual Networks](
#'     https://arxiv.org/abs/1603.05027) (CVPR 2016)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will
#' scale input pixels between `-1` and `1`.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet152v2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152V2>
#' @tether keras.applications.ResNet152V2
application_resnet152_v2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet152V2, args)
    set_preprocessing_attributes(model, keras$applications$resnet_v2)
}


#' Instantiates the ResNet50V2 architecture.
#'
#' @description
#'
#' # Reference
#' - [Identity Mappings in Deep Residual Networks](
#'     https://arxiv.org/abs/1603.05027) (CVPR 2016)
#'
#' For image classification use cases, see [this page for detailed examples](
#'     https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'     https://keras.io/guides/transfer_learning/).
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For ResNet, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model. [`application_preprocess_inputs()`] will
#' scale input pixels between `-1` and `1`.
#'
#' @returns
#' A Model instance.
#'
#' @param include_top
#' whether to include the fully-connected
#' layer at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet), or the path to the weights
#' file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified if `include_top`
#' is `FALSE` (otherwise the input shape has to be `(224, 224, 3)`
#' (with `"channels_last"` data format) or `(3, 224, 224)`
#' (with `"channels_first"` data format). It should have exactly 3
#' inputs channels, and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction when `include_top`
#' is `FALSE`.
#' - `NULL` means that the output of the model will be the 4D tensor
#'         output of the last convolutional block.
#' - `avg` means that global average pooling will be applied to the output
#'         of the last convolutional block, and thus the output of the
#'         model will be a 2D tensor.
#' - `max` means that global max pooling will be applied.
#'
#' @param classes
#' optional number of classes to classify images into, only to be
#' specified if `include_top` is `TRUE`, and if no `weights` argument is
#' specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top" layer.
#' When loading pretrained weights, `classifier_activation` can only
#' be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/resnet#resnet50v2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2>
#' @tether keras.applications.ResNet50V2
application_resnet50_v2 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$ResNet50V2, args)
    set_preprocessing_attributes(model, keras$applications$resnet_v2)
}


#' Instantiates the VGG16 model.
#'
#' @description
#'
#' # Reference
#' - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
#' https://arxiv.org/abs/1409.1556) (ICLR 2015)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' The default input size for this model is 224x224.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For VGG16, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model.
#' [`application_preprocess_inputs()`] will convert the input images from RGB to BGR,
#' then will zero-center each color channel with respect to the ImageNet
#' dataset, without scaling.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' whether to include the 3 fully-connected
#' layers at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)`
#' (with `channels_last` data format) or
#' `(3, 224, 224)` (with `"channels_first"` data format).
#' It should have exactly 3 input channels,
#' and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation`
#' can only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/vgg#vgg16-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16>
#' @tether keras.applications.VGG16
application_vgg16 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$VGG16, args)
    set_preprocessing_attributes(model, keras$applications$vgg16)
}


#' Instantiates the VGG19 model.
#'
#' @description
#'
#' # Reference
#' - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
#' https://arxiv.org/abs/1409.1556) (ICLR 2015)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' The default input size for this model is 224x224.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For VGG19, call [`application_preprocess_inputs()`] on your
#' inputs before passing them to the model.
#' [`application_preprocess_inputs()`] will convert the input images from RGB to BGR,
#' then will zero-center each color channel with respect to the ImageNet
#' dataset, without scaling.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' whether to include the 3 fully-connected
#' layers at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(224, 224, 3)`
#' (with `channels_last` data format) or
#' `(3, 224, 224)` (with `"channels_first"` data format).
#' It should have exactly 3 input channels,
#' and width and height should be no smaller than 32.
#' E.g. `(200, 200, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation` can
#' only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/vgg#vgg19-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19>
#' @tether keras.applications.VGG19
application_vgg19 <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer))
    model <- do.call(keras$applications$VGG19, args)
    set_preprocessing_attributes(model, keras$applications$vgg19)
}


#' Instantiates the Xception architecture.
#'
#' @description
#'
#' # Reference
#' - [Xception: Deep Learning with Depthwise Separable Convolutions](
#'     https://arxiv.org/abs/1610.02357) (CVPR 2017)
#'
#' For image classification use cases, see
#' [this page for detailed examples](
#'   https://keras.io/api/applications/#usage-examples-for-image-classification-models).
#'
#' For transfer learning use cases, make sure to read the
#' [guide to transfer learning & fine-tuning](
#'   https://keras.io/guides/transfer_learning/).
#'
#' The default input image size for this model is 299x299.
#'
#' # Note
#' Each Keras Application expects a specific kind of input preprocessing.
#' For Xception, call [`application_preprocess_inputs()`]
#' on your inputs before passing them to the model.
#' [`application_preprocess_inputs()`] will scale input pixels between `-1` and `1`.
#'
#' @returns
#' A model instance.
#'
#' @param include_top
#' whether to include the 3 fully-connected
#' layers at the top of the network.
#'
#' @param weights
#' one of `NULL` (random initialization),
#' `"imagenet"` (pre-training on ImageNet),
#' or the path to the weights file to be loaded.
#'
#' @param input_tensor
#' optional Keras tensor
#' (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#'
#' @param input_shape
#' optional shape tuple, only to be specified
#' if `include_top` is `FALSE` (otherwise the input shape
#' has to be `(299, 299, 3)`.
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 71.
#' E.g. `(150, 150, 3)` would be one valid value.
#'
#' @param pooling
#' Optional pooling mode for feature extraction
#' when `include_top` is `FALSE`.
#' - `NULL` means that the output of the model will be
#'     the 4D tensor output of the
#'     last convolutional block.
#' - `avg` means that global average pooling
#'     will be applied to the output of the
#'     last convolutional block, and thus
#'     the output of the model will be a 2D tensor.
#' - `max` means that global max pooling will
#'     be applied.
#'
#' @param classes
#' optional number of classes to classify images
#' into, only to be specified if `include_top` is `TRUE`, and
#' if no `weights` argument is specified.
#'
#' @param classifier_activation
#' A `str` or callable. The activation function to
#' use on the "top" layer. Ignored unless `include_top=TRUE`. Set
#' `classifier_activation=NULL` to return the logits of the "top"
#' layer.  When loading pretrained weights, `classifier_activation` can
#' only be `NULL` or `"softmax"`.
#'
#' @export
#' @seealso
#' + <https://keras.io/api/applications/xception#xception-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception>
#' @tether keras.applications.Xception
application_xception <-
function (include_top = TRUE, weights = "imagenet", input_tensor = NULL,
    input_shape = NULL, pooling = NULL, classes = 1000L, classifier_activation = "softmax")
{
    args <- capture_args(list(classes = as_integer, input_shape = normalize_shape))
    model <- do.call(keras$applications$Xception, args)
    set_preprocessing_attributes(model, keras$applications$xception)
}

#' Preprocessing and postprocessing utilities
#'
#' @description
#' These functions are used to preprocess and postprocess
#' inputs and outputs of Keras applications.
#'
#' @param model A Keras model initialized using any `application_` function.
#' @param x A batch of inputs to the model.
#' @param preds A batch of outputs from the model.
#' @param ... Additional arguments passed to the preprocessing or decoding function.
#' @param top The number of top predictions to return.
#' @param data_format
#' Optional data format of the image tensor/array.
#' `NULL` means the global setting
#' `config_image_data_format()` is used
#' (unless you changed it, it uses `"channels_last"`).
#' Defaults to `NULL`.
#'
#' @return
#' - A list of decoded predictions in case of `application_decode_predictions()`.
#' - A batch of preprocessed inputs in case of `application_preprocess_inputs()`.
#'
#' @examples \dontrun{
#' model <- application_convnext_tiny()
#'
#' inputs <- random_normal(c(32, 224, 224, 3))
#' processed_inputs <- application_preprocess_inputs(model, inputs)
#'
#' preds <- random_normal(c(32, 1000))
#' decoded_preds <- application_decode_predictions(model, preds)
#'
#' }
#' @name process_utils
NULL

#' @describeIn process_utils Pre-process inputs to be used in the model
#' @export
application_preprocess_inputs <- function(model, x, ..., data_format = NULL) {
  preprocess_input <- attr(model, "preprocess_input")
  if (is.null(preprocess_input)) not_found_errors()
  preprocess_input(x, data_format = data_format, ...)
}

#' @describeIn process_utils Decode predictions from the model
#' @export
application_decode_predictions <- function(model, preds, top = 5L, ...) {
  decode_predictions <- attr(model, "decode_predictions")
  if (is.null(decode_predictions)) not_found_errors()
  decode_predictions(preds, top = as_integer(top), ...)
}

not_found_errors <- function(model) {
  if (!inherits(model, "keras.src.models.model.Model")) {
    cli::cli_abort(c(
      x = "The {.arg model} argument must be a Keras model, got {.cls {head(class(model))}}"
    ))
  }

  if (model$name %in% list_model_names()) {
    cli::cli_abort(c(
      x = "The {.arg model} argument must be created using the `application_` functions.",
      i = "It looks like it was returned by a different type of call."
    ))
  }

  rlang::abort(c(x = "No preprocessing/decoding utilities found for this model."))
}

list_model_names <- function() {
  # this list is used to produce a nicer error message when a user initialized
  # the model using the raw interface instead of using the `application_` functions
  # it can be updated with something like:
  # model_names <- ls(envir = asNamespace("keras")) %>%
  #   purrr::keep(\(name) stringr::str_detect(name, "^application_")) %>%
  #   purrr::map_chr(\(name) do.call(name, list(weights = NULL))$name)
  # dput(model_names)
  c("convnext_base", "convnext_large", "convnext_small", "convnext_tiny",
    "convnext_xlarge", "densenet121", "densenet169", "densenet201",
    "efficientnetb0", "efficientnetb1", "efficientnetb2", "efficientnetb3",
    "efficientnetb4", "efficientnetb5", "efficientnetb6", "efficientnetb7",
    "efficientnetv2-b0", "efficientnetv2-b1", "efficientnetv2-b2",
    "efficientnetv2-b3", "efficientnetv2-l", "efficientnetv2-m",
    "efficientnetv2-s", "inception_resnet_v2", "inception_v3", "mobilenet_1.00_224",
    "mobilenetv2_1.00_224", "MobilenetV3large", "MobilenetV3small",
    "NASNet", "NASNet", "resnet101", "resnet101v2", "resnet152",
    "resnet152v2", "resnet50", "resnet50v2", "vgg16", "vgg19", "xception"
  )
}

set_preprocessing_attributes <- function(object, module) {
  attr(object, "preprocess_input") <- module$preprocess_input
  attr(object, "decode_predictions") <- module$decode_predictions
  object
}
