#' kms
#' 
#' A regression-style function call for Keras model sequential (KMS) which uses sparse matrices.
#' 
#' @param input_formula an object of class "formula" (or one coerceable to a formula): a symbolic description of the keras inputs. The outcome, y, is assumed to be categorical, e.g. "stars ~ mentions.tasty + mentions.fun".
#' @param data a data.frame.
#' @param keras_model_seq A compiled Keras sequential model. If non-NULL (NULL is the default), then bypasses the following `kms` parameters: layers, loss, metrics, and optimizer.
#' @param layers a list that creates a dense Keras model. Contains the number of units, activation type, and dropout rate. Example with three layers: layers = list(units = c(256, 128, NA), activation = c("relu", "relu", "softmax"), dropout = c(0.4, 0.3, NA)). If the final element of units is NA (default), set to the number of unique elements in y. See ?layer_dense or ?layer_dropout. 
#' @param pTraining Proportion of the data to be used for training the model;  0 < pTraining < 1. By default, pTraining == 0.8. Other observations used only postestimation (e.g., for confusion matrix). 
#' @param seed seed to passed to set.seed for partitioning data. If NULL (default), automatically generated.
#' @param validation_split Portion of data to be used for validating each epoch (i.e., portion of pTraining). To be passed to keras::fit. Default == 0.2. 
#' @param Nepochs Number of epochs. To be passed to keras::fit. Default == 25.  
#' @param batch_size To be passed to keras::fit. Default == 32. 
#' @param loss To be passed to keras::compile. Defaults to "binary_crossentropy" or "categorical_crossentropy" based on the number of distinct elements of y.
#' @param metrics To be passed to keras::compile. Default == c("accuracy").
#' @param optimizer To be passed to keras::compile. Default == "optimizer_rmsprop". 
#' @param ... Additional parameters to be passsed to Matrix::sparse.model.matrix.
#' @return kms_fit object. A list containing model, predictions, evaluations, as well as other details like how the data were split into testing and training.
#' @examples
#' # not run
#' # n <- 1000
#' # p <- 26
#' # X <- matrix(runif(n*p), ncol = p) 
#' # y <- letters[apply(X, 1, which.max)]
#' # DF <- data.frame(y, X)
#' # out <- kms("y ~ X", DF)
#' # out2 <- kms("y ~ . - 1",         # use X1 ... X26 but no intercept  
#' #             DF, pTraining = 0.9, Nepochs = 10, batch_size = 16)
#' # cars_out <- kms("mpg %/% 1 ~ grepl('Mazda', rownames(mtcars), ignore.case = TRUE)", mtcars)
#' @author Pete Mohanty
#' @importFrom Matrix sparse.model.matrix
#' @importFrom dplyr n_distinct
#' @export
kms <- function(input_formula, data, keras_model_seq = NULL, 
                 layers = list(units = c(128, NA), activation = c("relu", "softmax"),
                               dropout = c(0.4, NA)), 
                 pTraining = 0.8, seed = NULL, validation_split = 0.2, 
                 Nepochs = 25, batch_size = 32, loss = NULL, metrics = c("accuracy"),
                 optimizer = "optimizer_rmsprop", ...){
   
  if(pTraining < 0 || pTraining >= 1) 
    stop("pTraining, the proportion of data used for training, must be between 0 and 1.")
  
  form <- as.formula(input_formula)
  if(form[[1]] != "~" || length(form) != 3) 
    stop("Expecting formula of the form\n\ny ~ x1 + x2 + x3\n\nwhere y, x1, x2... are found in (the data.frame) data.")
  
  x_tmp <- sparse.model.matrix(form, data = data, ...)
  P <- ncol(x_tmp)
  N <- nrow(x_tmp)
  
  seed <- if(is.null(seed)) sample(range(.Random.seed), size = 1) else seed
  set.seed(seed)
  split <- sample(c("train", "test"), size = N, 
                  replace = TRUE, prob = c(pTraining, 1 - pTraining))
  
  x_train <- x_tmp[split == "train", ]
  x_test <- x_tmp[split == "test", ]
  remove(x_tmp)
  
  y <- eval(form[[2]], envir = data)
  n_distinct_y <- n_distinct(y)
  
  if(is.numeric(y)) 
    if((n_distinct_y == length(y) | min(y) < 0 | 
        sum(y %% 1) > length(y) * .Machine$double.eps))
      warning("y does not appear to be categorical.\n\n" )
  
  labs <- sort(unique(y))
  
  if(n_distinct_y > 2){
    
    y_cat <- to_categorical(match(y, labs) - 1) # make parameter y.categorical (??)
    # -1 for Python/C style indexing arrays, which starts at 0, must "undo"
    y_train <- y_cat[split == "train",]
    y_test <- y_cat[split == "test",]
    remove(y_cat)
    
  }else{
    
    y_train <- y[split == "train"]
    y_test <- y[split == "test"]
    
  }
  
  
  if(is.null(keras_model_seq)){
    
    if(is.na(layers$units[length(layers$units)]))
      layers$units[length(layers$units)] <- max(1, ncol(y_train))
    
    Nlayers <- length(layers$units)
    
    keras_model_seq <- keras_model_sequential() 
    for(i in 1:Nlayers){
      keras_model_seq <- if(i == 1){
        layer_dense(keras_model_seq, units = layers$units[i], activation = layers$activation[i], input_shape = c(P))
      }else{
        layer_dense(keras_model_seq, units = layers$units[i], activation = layers$activation[i])
      }
      if(i != Nlayers)
        model <- layer_dropout(keras_model_seq, rate = layers$rate[i])
    }
    
    if(is.null(loss)) 
      loss <- if(n_distinct(y) == 2) "binary_crossentropy" else "categorical_crossentropy" 
    
    keras_model_seq %>% compile(
      loss = loss,
      optimizer = do.call(optimizer, args = list()),
      metrics = metrics
    )
    
  }

  history <- keras_model_seq %>% fit(x_train, y_train, 
    epochs = Nepochs, 
    batch_size = batch_size, 
    validation_split = validation_split
  )
  
  evals <- keras_model_seq %>% evaluate(x_test, y_test)

  # 1 + to get back to R/Fortran land... 
  y_fit <- labs[1 + predict_classes(keras_model_seq, x_test)]

  object <- list(history = history, 
                 evaluations = evals, predictions = y_fit,
                 input_formula = input_formula, model = keras_model_seq, 
                 loss = loss, optimizer = optimizer, metrics = metrics,
                 N = N, P = P,
                 y_test = y[split == "test"],
                 seed = seed, split = split)
  
  object[["confusion"]] <- confusion(object)

  class(object) <- "kms_fit"
  return(object)
  
}




