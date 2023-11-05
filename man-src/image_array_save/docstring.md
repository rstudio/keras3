Saves an image stored as a NumPy array to a path or file object.

Args:
    path: Path or file object.
    x: NumPy array.
    data_format: Image data format, either `"channels_first"` or
        `"channels_last"`.
    file_format: Optional file format override. If omitted, the format to
        use is determined from the filename extension. If a file object was
        used instead of a filename, this parameter should always be used.
    scale: Whether to rescale image values to be within `[0, 255]`.
    **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
