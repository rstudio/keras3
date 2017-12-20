foRmula keRas
================
Pete Mohanty
December 19, 2017

The goal of this document is to introduce `lstm`, a regression style function which allows users to call `keras` with `R` `formula` objects and which splits training and test data into sparse matrices. Let's start with an example using `rtweet` from `@kearneymw`. The examples here don't provide particularly predictive models so much as show how using `formula` objects can smooth data cleaning and hyperparameter selection.

``` r
library(rtweet)
rt <- search_tweets(
  "#rstats", n = 10000, include_rts = FALSE
)
dim(rt)
```

    [1] 2476   42

``` r
summary(rt$retweet_count)
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      0.000   0.000   0.000   3.468   2.000 327.000 

Suppose we wanted to predict how many times a tweet with `#rstat` is going to be retweeted. And suppose we wanted to bin the retweent count into five categories (none, 1-10, 11-50, 51-99, and 100 or more). Suppose we believe that the twitter handle matters as does day and time of day. Also, this demo makes use of a `grepv`, a wrapper function for grep that returns an `N` length dummy vector.

``` r
library(keras)
breaks <- c(-1, 0, 1, 10, 50, 100, 10000)
out <- lstm("cut(retweet_count, breaks) ~ screen_name + 
            grepv('gg', text) + grepv('tidy', text) + 
            grepv('rstudio', text, ignore.case = TRUE) + grepv('python', text, ignore.case = TRUE) + 
            grepv('cran', text, ignore.case = TRUE) +
            grepv('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')", data = rt)
plot(out$history)
```

![](foRmula_keRas_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-3-1.png)

``` r
summary(out$model)
```

    ___________________________________________________________________________
    Layer (type)                     Output Shape                  Param #     
    ===========================================================================
    dense_1 (Dense)                  (None, 128)                   138624      
    ___________________________________________________________________________
    dropout_1 (Dropout)              (None, 128)                   0           
    ___________________________________________________________________________
    dense_2 (Dense)                  (None, 6)                     774         
    ===========================================================================
    Total params: 139,398
    Trainable params: 139,398
    Non-trainable params: 0
    ___________________________________________________________________________

``` r
out$confusion
```

                 
                  (-1,0] (0,1] (1,10] (10,50] (50,100] (100,1e+04]
      (-1,0]         224    21     31       0        0           0
      (0,1]           36     8     19       1        0           0
      (1,10]          49     7     45       9        0           0
      (10,50]          8     2     16       8        0           0
      (50,100]         3     0      2       2        0           0
      (100,1e+04]      1     0      0       0        0           0

``` r
out$evaluations
```

    $loss
    [1] 1.279366

    $acc
    [1] 0.5792683

Let's say we want to add some data about how many other people are mentioned in each tweet and switch to a (discretized) log scale.

``` r
rt$Nmentions <- unlist(lapply(rt$mentions_screen_name, 
                              function(x){length(x[[1]]) - is.na(x[[1]])}))

out2 <- lstm("floor(log(retweet_count + 1)) ~ Nmentions + screen_name + 
            grepv('gg', text) + grepv('tidy', text) + 
            grepv('rstudio', text, ignore.case = TRUE) + grepv('python', text, ignore.case = TRUE) + 
            grepv('cran', text, ignore.case = TRUE) +
            grepv('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')", 
            data = rt, Nepochs = 10)
out2$evaluations
```

    $loss
    [1] 0.7304347

    $acc
    [1] 0.743487

``` r
out2$confusion
```

       
          0   1   2
      0 342  13   2
      1  67  14   8
      2  18   4  15
      3   3   5   5
      4   2   0   1

Heading in the right direction. Suppose instead we wanted to add who was mentioned.

``` r
input.formula <- "floor(log(retweet_count + 1)) ~ Nmentions + screen_name + 
            grepv('gg', text) + grepv('tidy', text) + 
            grepv('rstudio', text, ignore.case = TRUE) + grepv('python', text, ignore.case = TRUE) + 
            grepv('cran', text, ignore.case = TRUE) +
            grepv('trump', text, ignore.case = TRUE) +
            weekdays(rt$created_at) + format(rt$created_at, '%d') + 
            format(rt$created_at, '%H')"

handles <- names(table(unlist(rt$mentions_screen_name)))

for(i in 1:length(handles)){
  lab <- paste0("mentions_", handles[i])
  rt[[lab]] <- grepv(handles[i], rt$mentions_screen_name)
  input.formula <- paste(input.formula, "+", lab)
}

out3 <- lstm(input.formula, data = rt, Nepochs = 10)
out3$evaluations
```

    $loss
    [1] 0.7727613

    $acc
    [1] 0.7126866

``` r
out3$confusion
```

       
          0   1   2
      0 344  23   2
      1  64  20  15
      2  21   8  18
      3   8   3   6
      4   3   0   1

Marginal improvement but the model is still clearly overpredicting the modal outcome (zero retweets) and struggling to forecast the rare, popular tweets. Maybe the model needs more layers.

``` r
out4 <- lstm(input.formula, data = rt, 
            layers = list(units = c(405, 135, 45, 15, NA), 
                         activation = c("softmax", "relu", "relu", "relu", "softmax"), 
                         dropout = c(0.7, 0.6, 0.5, 0.4, NA)),
            Nepochs = 6)
out4$evaluations
```

    $loss
    [1] 0.8312168

    $acc
    [1] 0.7256461

``` r
out4$confusion
```

       
          0
      0 365
      1  90
      2  31
      3  12
      4   4
      5   1

Suppose we wanted to see if the estimates were stable across 10 test/train splits.

``` r
est <- list()
accuracy <- c()
for(i in 1:10){
  est[[paste0("seed", i)]] <- lstm(input.formula, data = rt, seed = i,
            layers = list(units = c(405, 135, 45, 15, NA), 
                         activation = c("softmax", "relu", "relu", "relu", "softmax"), 
                         dropout = c(0.7, 0.6, 0.5, 0.4, NA)),
            Nepochs = 10)
  accuracy[i] <- est[[paste0("seed", i)]][["evaluations"]][["acc"]]
}
accuracy
```

     [1] 0.6993988 0.7250922 0.6993865 0.7062500 0.7137255 0.6946108 0.7080868
     [8] 0.7248996 0.7278481 0.7464789

Hmmm... Maybe Model 3 is the closest ... Or maybe we just need more data :)

Though `lstm` contains a number of parameters, the goal is not to replace all the vast customizability that `keras` offers. Rather, like `qplot` in the `ggplot` library, `lstm` offers convenience for common scenarios. Or, perhaps better, like `MCMCpack` or `rstan` do for Bayesian MCMC, `lstm` aims to introduce users familiar with regression in R to neural nets without steep scripting stumbling blocks. Suggestions are more than welcome!
