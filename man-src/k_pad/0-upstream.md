keras.ops.pad
__signature__
(x, pad_width, mode='constant')
__doc__
Pad a tensor.

Args:
    x: Tensor to pad.
    pad_width: Number of values padded to the edges of each axis.
        `((before_1, after_1), ...(before_N, after_N))` unique pad
        widths for each axis.
        `((before, after),)` yields same before and after pad for
        each axis.
        `(pad,)` or `int` is a shortcut for `before = after = pad`
        width for all axes.
    mode: One of `"constant"`, `"edge"`, `"linear_ramp"`,
        `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
        `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
        `"circular"`. Defaults to`"constant"`.

Note:
    Torch backend only supports modes `"constant"`, `"reflect"`,
    `"symmetric"` and `"circular"`.
    Only Torch backend supports `"circular"` mode.

Note:
    Tensorflow backend only supports modes `"constant"`, `"reflect"`
    and `"symmetric"`.

Returns:
    Padded tensor.
