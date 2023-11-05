Move axes of a tensor to new positions.

Other axes remain in their original order.

Args:
    x: Tensor whose axes should be reordered.
    source: Original positions of the axes to move. These must be unique.
    destination: Destinations positions for each of the original axes.
        These must also be unique.

Returns:
    Tensor with moved axes.
