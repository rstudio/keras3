# Define a custom `LearningRateSchedule` class

Subclass the Keras `LearningRateSchedule` base class.

You can use a learning rate schedule to modulate how the learning rate
of your optimizer changes over time.

Several built-in learning rate schedules are available, such as
[`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_exponential_decay.md)
or
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_piecewise_constant_decay.md):

    lr_schedule <- learning_rate_schedule_exponential_decay(
      initial_learning_rate = 1e-2,
      decay_steps = 10000,
      decay_rate = 0.9
    )
    optimizer <- optimizer_sgd(learning_rate = lr_schedule)

A `LearningRateSchedule()` instance can be passed in as the
`learning_rate` argument of any optimizer.

To implement your own schedule object, you should implement the `call`
method, which takes a `step` argument (a scalar integer backend tensor,
the current training step count). Note that `step` is 0-based (i.e., the
first step is `0`). Like for any other Keras object, you can also
optionally make your object serializable by implementing the
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
and
[`from_config()`](https://keras3.posit.co/dev/reference/get_config.md)
methods.

## Usage

``` r
LearningRateSchedule(
  classname,
  call = NULL,
  initialize = NULL,
  get_config = NULL,
  ...,
  public = list(),
  private = list(),
  inherit = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  String, the name of the custom class. (Conventionally, CamelCase).

- call, initialize, get_config:

  Recommended methods to implement. See description and details
  sections.

- ..., public:

  Additional methods or public members of the custom class.

- private:

  Named list of R objects (typically, functions) to include in instance
  private environments. `private` methods will have all the same symbols
  in scope as public methods (See section "Symbols in Scope"). Each
  instance will have it's own `private` environment. Any objects in
  `private` will be invisible from the Keras framework and the Python
  runtime.

- inherit:

  What the custom class will subclass. By default, the base keras class.

- parent_env:

  The R environment that all class methods will have as a grandparent.

## Value

A function that returns `LearningRateSchedule` instances, similar to the
built-in `learning_rate_schedule_*` family of functions.

## Example

    my_custom_learning_rate_schedule <- LearningRateSchedule(
      classname = "MyLRSchedule",

      initialize = function(initial_learning_rate) {
        self$initial_learning_rate <- initial_learning_rate
      },

      call = function(step) {
        # note that `step` is a tensor
        # and call() will be traced via tf_function() or similar.

        str(step) # <KerasVariable shape=(), dtype=int64, path=SGD/iteration>

        # print 'step' every 1000 steps
        op_cond((step %% 1000) == 0,
                \() {tensorflow::tf$print(step); NULL},
                \() {NULL})
        self$initial_learning_rate / (step + 1)
      }
    )

    optimizer <- optimizer_sgd(
      learning_rate = my_custom_learning_rate_schedule(0.1)
    )

    # You can also call schedule instances directly
    # (e.g., for interactive testing, or if implementing a custom optimizer)
    schedule <- my_custom_learning_rate_schedule(0.1)
    step <- keras$Variable(initializer = op_ones,
                           shape = shape(),
                           dtype = "int64")
    schedule(step)

    ## <Variable path=variable, shape=(), dtype=int64, value=1>

    ## tf.Tensor(0.0, shape=(), dtype=float64)

## Methods available:

- get_config()

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

Other optimizer learning rate schedules:  
[`learning_rate_schedule_cosine_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_cosine_decay.md)  
[`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_cosine_decay_restarts.md)  
[`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_exponential_decay.md)  
[`learning_rate_schedule_inverse_time_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_inverse_time_decay.md)  
[`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_piecewise_constant_decay.md)  
[`learning_rate_schedule_polynomial_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_polynomial_decay.md)  
