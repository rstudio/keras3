Return elements chosen from `x1` or `x2` depending on `condition`.

Args:
    condition: Where `True`, yield `x1`, otherwise yield `x2`.
    x1: Values from which to choose when `condition` is `True`.
    x2: Values from which to choose when `condition` is `False`.

Returns:
    A tensor with elements from `x1` where `condition` is `True`, and
    elements from `x2` where `condition` is `False`.
