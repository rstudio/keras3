foRmula keRas
================
Pete Mohanty
December 20, 2017

The goal of this document is to introduce `ksm` (as in `keras_sequential_model()`), a regression-style function which allows users to call `keras` neural nets with `R` `formula` objects. `ksm` splits training and test data into sparse matrices and contains the major parameters found in `library(keras)` as inputs (loss function, batch size, number of epochs, etc.). To install my [branch](https://github.com/rdrr1990/keras) of `keras`,

``` r
devtools::install_github("rdrr1990/keras")
```

Let's start with an example using `rtweet` (from `@kearneymw`). The examples here don't provide particularly predictive models so much as show how using `formula` objects can smooth data cleaning and hyperparameter selection.

``` r
library(rtweet)
rt <- search_tweets("#rstats", n = 5000, include_rts = FALSE)
dim(rt)
```

    [1] 2571   42

``` r
summary(rt$retweet_count)
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      0.000   0.000   0.000   3.494   2.000 327.000 

Suppose we wanted to predict how many times a tweet with `#rstat` is going to be retweeted. And suppose we wanted to bin the retweent count into five categories (none, 1-10, 11-50, 51-99, and 100 or more). Suppose we believe that the twitter handle and source matters as does day of week and time of day.

``` r
library(keras)
breaks <- c(-1, 0, 1, 10, 50, 100, 10000)
out <- ksm("cut(retweet_count, breaks) ~ screen_name + source +
            grepl('gg', text) + grepl('tidy', text) + 
            grepl('rstudio', text, ignore.case = TRUE) +
            grepl('cran', text, ignore.case = TRUE) +
            grepl('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + 
            format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')", data = rt)
plot(out$history)
```

![](foRmula_keRas_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-4-1.png)

``` r
summary(out$model)
```

    ___________________________________________________________________________
    Layer (type)                     Output Shape                  Param #     
    ===========================================================================
    dense_1 (Dense)                  (None, 128)                   149376      
    ___________________________________________________________________________
    dropout_1 (Dropout)              (None, 128)                   0           
    ___________________________________________________________________________
    dense_2 (Dense)                  (None, 6)                     774         
    ===========================================================================
    Total params: 150,150
    Trainable params: 150,150
    Non-trainable params: 0
    ___________________________________________________________________________

``` r
out$confusion
```

                 
                  (-1,0] (0,1] (1,10] (10,50] (50,100] (100,1e+04]
      (-1,0]         221    38     32       4        0           0
      (0,1]           35    14     13       1        0           0
      (1,10]          36    17     45      14        0           0
      (10,50]          8     3     17      11        0           0
      (50,100]         1     0      1       1        0           0
      (100,1e+04]      1     0      0       1        0           0

``` r
out$evaluations
```

    $loss
    [1] 1.300226

    $acc
    [1] 0.5661479

Let's say we want to add some data about how many other people are mentioned in each tweet and switch to a (discretized) log scale.

``` r
rt$Nmentions <- unlist(lapply(rt$mentions_screen_name, 
                              function(x){length(x[[1]]) - is.na(x[[1]])}))

out2 <- ksm("floor(log(retweet_count + 1)) ~ Nmentions + screen_name + source +
            grepl('gg', text) + grepl('tidy', text) + 
            grepl('rstudio', text, ignore.case = TRUE) +
            grepl('cran', text, ignore.case = TRUE) +
            grepl('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + 
            format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')", 
            data = rt, Nepochs = 10)
out2$evaluations
```

    $loss
    [1] 0.7796955

    $acc
    [1] 0.7351129

``` r
out2$confusion
```

       
          0   1   2
      0 336  14   1
      1  59  10   4
      2  31   1  12
      3  11   2   3
      4   1   0   2

Heading in the right direction. Suppose instead we wanted to add who was mentioned.

``` r
input.formula <- "floor(log(retweet_count + 1)) ~ Nmentions + screen_name + source +
            grepl('gg', text) + grepl('tidy', text) + 
            grepl('rstudio', text, ignore.case = TRUE) + grepl('python', text, ignore.case = TRUE) + 
            grepl('cran', text, ignore.case = TRUE) +
            grepl('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')"

handles <- names(table(unlist(rt$mentions_screen_name)))

for(i in 1:length(handles)){
  lab <- paste0("mentions_", handles[i])
  rt[[lab]] <- grepl(handles[i], rt$mentions_screen_name)
  input.formula <- paste(input.formula, "+", lab)
}

out3 <- ksm(input.formula, data = rt, Nepochs = 10)
out3$evaluations
```

    $loss
    [1] 0.8192519

    $acc
    [1] 0.7142857

``` r
out3$confusion
```

       
          0   1   2
      0 326  28   0
      1  50  18   5
      2  22  13  16
      3   5   7   8
      4   2   3   1

Marginal improvement but the model is still clearly overpredicting the modal outcome (zero retweets) and struggling to forecast the rare, popular tweets. Maybe the model needs more layers.

``` r
out4 <- ksm(input.formula, data = rt, 
            layers = list(units = c(405, 135, 45, 15, NA), 
                         activation = c("softmax", "relu", "relu", "relu", "softmax"), 
                         dropout = c(0.7, 0.6, 0.5, 0.4, NA)),
            Nepochs = 6)
out4$evaluations
```

    $loss
    [1] 0.8339834

    $acc
    [1] 0.7028037

``` r
out4$confusion
```

       
          0
      0 376
      1  86
      2  54
      3  14
      4   5

Suppose we wanted to see if the estimates were stable across 10 test/train splits.

``` r
est <- list()
accuracy <- c()
for(i in 1:10){
  est[[paste0("seed", i)]] <- ksm(input.formula, data = rt, seed = i,
            layers = list(units = c(405, 135, 45, 15, NA), 
                         activation = c("softmax", "relu", "relu", "relu", "softmax"), 
                         dropout = c(0.7, 0.6, 0.5, 0.4, NA)),
            Nepochs = 10)
  accuracy[i] <- est[[paste0("seed", i)]][["evaluations"]][["acc"]]
}
accuracy
```

     [1] 0.7312860 0.6983842 0.7312253 0.7236581 0.7293233 0.7255278 0.7437859
     [8] 0.7123552 0.7318548 0.7153846

Hmmm... Maybe Model 3 is the closest ... Or maybe we just need more data :)

Though `ksm` contains a number of parameters, the goal is not to replace all the vast customizability that `keras` offers. Rather, like `qplot` in the `ggplot` library, `ksm` offers convenience for common scenarios. Or, perhaps better, like `MCMCpack` or `rstan` do for Bayesian MCMC, `ksm` aims to introduce users familiar with regression in R to neural nets without steep scripting stumbling blocks. Next on the to-do list is to add support for accepting keras models as a parameter. Suggestions are more than welcome!
