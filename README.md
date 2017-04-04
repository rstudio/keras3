R interface to Keras
================

[![Build Status](https://travis-ci.org/rstudio/keras.svg?branch=master)](https://travis-ci.org/rstudio/keras) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/rstudio/keras/blob/master/LICENSE)

Keras is a high-level neural networks API developed with a focus on enabling fast experimentation. The R interface to Keras uses [TensorFlow](https://rstudio.github.io/tensorflow/) as it's underlying computation engine.

*Being able to go from idea to result with the least possible delay is key to doing good research.* Use Keras if you need a deep learning library that:

-   Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
-   Supports both convolutional networks and recurrent networks, as well as combinations of the two.
-   Runs seamlessly on CPU and GPU.

Visit the home page for the Keras project at <https://keras.io/>

Read the documentation for the R package at <https://rstudio.github.io/keras>

Guiding principles
------------------

-   **User friendliness**. Keras is an API designed for human beings, not machines. It puts user experience front and center. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

-   **Modularity**. A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

-   **Easy extensibility**. New modules are simple to add (as new functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

-   **Work with R**. No separate model configuration files in a declarative format. Models are described in R code, which is compact, easier to debug, and allows for ease of extensibility.

Getting started: 30 seconds to Keras
------------------------------------

The core data structure of Keras is a *model*, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers. For more complex architectures, you should use the Keras functional API, which allows to build arbitrary graphs of layers.

Here is the Sequential model:

``` r
library(keras)

model <- model_sequential() %>% 
  layer_dense(units = 64, input_shape = 100) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 10) %>% 
  layer_activation(activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(lr = 0.02),
    metrics = c('accuracy')
  )
```

You can now iterate on your training data in batches (`x_train` and `y_train` are R matrices):

``` r
fit(model, x_train, y_train, epochs = 5, batch_size = 32)
```

Evaluate your performance in one line:

``` r
loss_and_metrics <- evaluate(model, x_test, y_test, batch_size = 128)
```

Or generate predictions on new data:

``` r
classes <- predict(model, x_test, batch_size = 128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

-   [Getting started with the Sequential model](articles/sequential_model.html)

-   [Getting started with the functional API](articles/functional_api.html)

Within the [examples](articles/examples) you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.

Installation
------------

In order to use Keras you should first install TensorFlow (version 1.1 or higher). Instructions for installing TensorFlow are here: <https://www.tensorflow.org/install/>.

Then, install the Keras R package from GitHub:

``` r
devtools::install_github("rstudio/keras")
```

Why this name, Keras?
---------------------

Keras (κέρας) means horn in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the Odyssey, where dream spirits (Oneiroi, singular Oneiros) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

> "Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them." Homer, Odyssey 19. 562 ff (Shewring translation).
