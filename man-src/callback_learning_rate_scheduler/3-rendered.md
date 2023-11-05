Learning rate scheduler.

@description
At the beginning of every epoch, this callback gets the updated learning
rate value from `schedule` function provided at `__init__`, with the current
epoch and current learning rate, and applies the updated learning rate on
the optimizer.

# Examples

```r
# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
scheduler <- function(epoch, lr) {
    if (epoch < 10) {
        return(lr)
    } else {
        return(lr * exp(-0.1))
    }
}

model <- keras_model_sequential() %>%
  layer_dense(units = 10)
model %>% compile(optimizer = optimizer_sgd(), loss = 'mse')
round(model$optimizer$lr, 5)
```

```
## 'SGD' object has no attribute 'lr'
```


```r
callback <- callback_learning_rate_scheduler(schedule = scheduler)
history <- model %>% fit(x = matrix(runif(100), nrow = 5, ncol = 20),
                         y = rep(0, 5),
                         epochs = 15, callbacks = list(callback), verbose = 0)
```

```
## Graph execution error:
## 
## Detected at node compile_loss/mse/sub defined at (most recent call last):
## <stack traces unavailable>
## Incompatible shapes: [5] vs. [5,10]
## 	 [[{{node compile_loss/mse/sub}}]]
## 	tf2xla conversion failed while converting __inference_one_step_on_data_1648[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
## 	 [[StatefulPartitionedCall]] [Op:__inference_one_step_on_iterator_1663]
```

```r
round(model$optimizer$lr, 5)
```

```
## 'SGD' object has no attribute 'lr'
```

@param schedule A function that takes an epoch index (integer, indexed from 0)
    and current learning rate (float) as inputs and returns a new
    learning rate as output (float).
@param verbose Integer. 0: quiet, 1: log update messages.

@export
@family callback
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler>
