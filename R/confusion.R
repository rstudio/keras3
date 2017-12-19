#' confusion
#' 
#' Confusion matrix or (for larger number of levels) confusion table.
#' 
#' @param object Optional fit object. confusion() assumes object contains holdout/vaidation data as `y_test` and the forecasts/classifications as `predictions` but alternative variable names can be specified with the input arguments by those names.
#' @param y_test A vector of holdout/validation data or the name in object (if fit object provided but alternative variable name required).
#' @param predictions A vector predictions or the name in object (if fit object provided but alternative variable name required).
#' @param return_xtab Logical. If TRUE, returns confusion matrix, which is a crosstable with correct predictions on the diagonal (if all levels are predicted at least once). If FALSE, returns (rectangular) table with columns for percent correct, most common misclassification, second most common misclassification, and other predictions. Defaults to TRUE (crosstable-style) only if number of levels < 6.
#'  
#' @return confusion matrix or table as specified by return_xtab.
#' @export
confusion <- function(y_test = NULL, predictions = NULL, object = NULL, return_xtab = NULL){
  
  obj <- data.frame(y_test = if(is.null(object)) y_test else object[[if(is.null(y_test)) "y_test" else y_test]],
                    predictions = if(is.null(object)) predictions else object[[if(is.null(predictions)) "predictions" else predictions]],
                    stringsAsFactors = FALSE)

  return_xtab <- if(is.null(return_xtab)) n_distinct(obj$predictions) < 6 else return_xtab 
  
  if(return_xtab){
    
    cf <- table(obj$y_test, obj$predictions)
    return(cf)
    
  }else{
    
    N <- nrow(obj)
    obj <- obj %>% mutate(correct = y_test == predictions)
    cf <- group_by(obj, y_test) %>% summarise(pCorrect = mean(correct)) %>% as.data.frame 
    # confusion 
    
    cf[["MCE"]] <- NA # Most Common Error
    cf[["pMCE"]] <- 0 # proportion that are MCE
    cf[["MCE2"]] <- NA # second most common error
    cf[["pMCE2"]] <- 0 
    cf[["pOther"]] <- 0
    
    for(i in 1:nrow(cf)){
      
      wrong <- obj %>% filter(!correct, y_test == cf$y_test[i]) %>% 
        group_by(predictions) %>% summarise(Nwrong = sum(!correct))
      wrong <- wrong[order(wrong$Nwrong, decreasing = TRUE), ]
      
      if(nrow(wrong) > 0){
        
        cf$MCE[i] <- wrong$predictions[1]
        cf$pMCE[i] <- wrong$Nwrong[1]/N
        
        if(nrow(wrong) > 1){
          
          cf$MCE2[i] <- wrong$predictions[2]
          cf$pMCE2[i] <- wrong$Nwrong[2]/N
          
        }
      }
      cf$pOther[i] <- 1 - (cf$pCorrect[i] + cf$pMCE[i] + cf$pMCE2[i])
    }
   return(cf) 
  }
    
}  
    
grepv <- function(pattern, x){
  z <- vector(mode = "integer", length = length(x))
  z[grep(pattern, x)] <- 1
  return(z)
}

