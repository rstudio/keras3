Loads an image into PIL format.

@description

# Usage
```python
image = keras.utils.load_img(image_path)
input_arr = keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
```

@returns
    A PIL Image instance.

@param path Path to image file.
@param color_mode One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
    The desired image format.
@param target_size Either `None` (default to original size) or tuple of ints
    `(img_height, img_width)`.
@param interpolation Interpolation method used to resample the image if the
    target size is different from that of the loaded image. Supported
    methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
    If PIL version 1.1.3 or newer is installed, `"lanczos"`
    is also supported. If PIL version 3.4.0 or newer is installed,
    `"box"` and `"hamming"` are also
    supported. By default, `"nearest"` is used.
@param keep_aspect_ratio Boolean, whether to resize images to a target
    size without aspect ratio distortion. The image is cropped in
    the center with target aspect ratio before resizing.

@export
@family utils
@seealso
+ <https:/keras.io/api/data_loading/image#loadimg-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img>
