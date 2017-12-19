#' lstm
#' 
#' A regression-style function call for LSTM for Keras for R which uses sparse matrices.
#' 
#' @param input.formula an object of class "formula" (or one coerceable to a formula): a symbolic description of the keras inputs. The outcome, y, is assumed to be categorical, e.g. "stars ~ mentions.tasty + mentions.fun".
#' @param data a data.frame.
#' @param layers a list that contains the number of units, activation type, and dropout rate. Example with three layers and length(unique(y)) == 10: layers = list(units = c(256, 128, 10), activation = c("relu", "relu", "softmax"), dropout = c(0.4, 0.3, NA)). If the final element of units is NA (the default), chosen to be the number of unique elements in y. See ?layer_dense or ?layer_dropout. 
#' @param pTraining Proportion of the data to be used for training the model;  0 < pTraining < 1. By default, pTraining == 0.8. Other observations used only postestimation (e.g., for confusion matrix). 
#' @param seed seed to passed to set.seed for partitioning data. If NULL (default), automatically generated.
#' @param validation_split Portion of data to be used for validating each epoch (i.e., portion of pTraining). To be passed to keras::fit. Default == 0.2. 
#' @param Nepochs Number of epochs. To be passed to keras::fit. Default == 25.  
#' @param batch_size To be passed to keras::fit. Default == 32. 
#' @param loss To be passed to keras::compile. Default == "categorical_crossentropy".
#' @param metrics To be passed to keras::compile. Default == c("accuracy").
#' @param optimizer To be passed to keras::compile. Default == "optimizer_rmsprop". 
#' @param ... Additional parameters to be passsed to Matrix::sparse.model.matrix.
#' @return keras.fit object. A list containing model, predictions, evaluations, as well as details on the function call.
#' @examples
#' n <- 1000
#' p <- 26
#' X <- matrix(runif(n*p), ncol = p) 
#' y <- letters[apply(X, 1, which.max)]
#' DF <- data.frame(y, X)
#' out <- lstm("y ~ X", DF)
#' @author Pete Mohanty
#' @importFrom Matrix sparse.model.matrix
#' @importFrom dplyr n_distinct mutate group_by summarise select 
#' @export
lstm <- function(input.formula, data, 
                 layers = list(units = c(128, NA), activation = c("relu", "softmax"), dropout = c(0.4, NA)), 
                 pTraining = 0.8, seed = NULL, validation_split = 0.2, 
                 Nepochs = 25, batch_size = 32, loss = "categorical_crossentropy", metrics = c("accuracy"),
                 optimizer = "optimizer_rmsprop", ...){
   
  if(pTraining < 0 || pTraining >= 1) 
    stop("pTraining, the proportion of data used for training, must be between 0 and 1.")
  
  form <- as.formula(input.formula)
  if(form[[1]] != "~" || length(form) != 3) 
    stop("Expecting formula of the form\n\ny ~ x1 + x2 + x3\n\nwhere y, x1, x2... are found in (the data.frame) data.")
    
  N <- nrow(data)
  
  seed <- if(is.null(seed)) sample(range(.Random.seed), size = 1) else seed
  set.seed(seed)
  split <- sample(c("train", "test"), size = N, 
                  replace = TRUE, prob = c(pTraining, 1 - pTraining))
  
  x_tmp <- sparse.model.matrix(form, data = data, ...) # drop intercept?
  P <- ncol(x_tmp)
  
  x_train <- x_tmp[split == "train", ]
  x_test <- x_tmp[split == "test", ]
  remove(x_tmp)
  
  y <- eval(form[[2]], envir = data)
   
  if(is.numeric(y)) 
    if((n_distinct(y) == length(y) | min(y) < 0 | sum(y %% 1) < length(y) * .Machine$double.eps))
      warning("y does not appear to be categorical.\n\n" )
  
  if(is.na(layers$units[length(layers$units)]))
    layers$units[length(layers$units)] <- n_distinct(y)
  
  labs <- sort(unique(y))
  y_cat <- to_categorical(match(y, labs) - 1) # make parameter y.categorical (??)
  # -1 for Python/C style indexing arrays, which starts at 0, must "undo"
  y_train <- y_cat[split == "train",]
  y_test <- y_cat[split == "test",]
  remove(y_cat)
  
  Nlayers <- length(layers$units)
  model <- keras_model_sequential() 
  for(i in 1:Nlayers){
    model <- if(i == 1){
      layer_dense(model, units = layers$units[i], activation = layers$activation[i], input_shape = c(P))
    }else{
      layer_dense(model, units = layers$units[i], activation = layers$activation[i])
    }
    if(i != Nlayers)
      model <- layer_dropout(model, rate = layers$rate[i])
  }
  summary(model)
  
  model %>% compile(
    loss = loss,
    optimizer = do.call(optimizer, args = list()),
    metrics = metrics
  )
  
  history <- model %>% fit(
    x_train, y_train, 
    epochs = Nepochs, 
    batch_size = batch_size, 
    validation_split = validation_split
  )
  
  evals <- model %>% evaluate(x_test, y_test)
  print(evals)
  
  # 1 + to get back to R/Fortran land... 
  y_fit <- labs[1 + predict_classes(model, x_test)]

  object <- list(input.formula = input.formula, model = model, history = history, 
                 evaluations = evals, predictions = y_fit, 
                 y_test = y[split=="test"],
                 layers = layers, seed = seed, split = split)
  object[["confusion"]] <- confusion(object)
  class(object) <- "keras.fit"
  return(object)
  
}

#' @export    
grepv <- function(pattern, x){
  z <- vector(mode = "integer", length = length(x))
  z[grep(pattern, x)] <- 1
  return(z)
}



