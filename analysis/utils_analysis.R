# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

# This script provides utility functions for analysing the results.
# It also sets the visual characsteristics of the plot as specified in the Excel sheet visual_params.xlsx

#### VISUAL PARAMETERS ####
vis_features <- openxlsx::read.xlsx("visual_params.xlsx", sheet = 1)
vis_models <- openxlsx::read.xlsx("visual_params.xlsx", sheet = 2)

features_names = vis_features$feature; names(features_names) <- features_names
features_names_print <- vis_features$name; names(features_names_print) <- features_names
features_col <- vis_features$color; names(features_col) <- features_names
features_pch <- vis_features$pch; names(features_pch) <- features_names
features_plot <- vis_features$plot == 1; names(features_plot) <- features_names

algos_names <- vis_models$algorithm; names(algos_names) <- algos_names
algos_names_print <- vis_models$name; names(algos_names_print) <- algos_names
algos_col <- vis_models$color; names(algos_col) <- algos_names
algos_pch <- vis_models$pch; names(algos_pch) <- algos_names
algos_lty <- vis_models$lty; names(algos_lty) <- algos_names
algos_plot <- vis_models$plot == 1; names(algos_plot) <- algos_names


country_spec <- openxlsx::read.xlsx("visual_params.xlsx", sheet = 3, colNames = F)
country_names <- country_spec[,1]
country_names_print <- country_spec[,2]
names(country_names_print) <- country_names



#### FUNCTIONS ####

measurePerf <- function(criterion,predicted,threshold = .5, random = F, weights = c(1,1)){
  # This function computes an array of performance metrics 
   
  # :param numeric vector criterion: observed class labels
  # :param numeric vector predicted: predicted probability of the positive class
  # :param double threshold: values on the "predicted vector" are are assigned to 
  #   the positive class if their value is greater than the threshold 
  # :param boolean random: If true, values with a predicted probability of 0.5
      #are randomly assigned to one of the claases
  #: param numeric vector weights of size 2: First value weights objects in the positive class,
  #   second value weights objects in the negative class.
  ix <- !is.na(predicted)
  criterion <- criterion[ix]
  predicted <- predicted[ix]
  
  predicted.t <- ifelse(predicted > threshold,1,0)
  if(random & threshold == .5)
    predicted.t[predicted==threshold] <- round(stats::runif(sum(predicted==threshold)))
  
  if(length(criterion) >1 && stats::sd(criterion) > 0){
    auc <- calcAUC(predicted,criterion)[1]
  } else auc <- NA
  tp <- as.numeric(sum(predicted.t==1&criterion==1)) * weights[1]
  fp <- as.numeric(sum(predicted.t==1&criterion==0)) * weights[2]
  tn <- as.numeric(sum(predicted.t==0&criterion==0)) * weights[2]
  fn <- as.numeric(sum(predicted.t==0&criterion==1)) * weights[1]
  accuracy <- (tp+tn)/(tp+fp+tn+fn)
  tp.rate <- tp/(tp+fn)
  fp.rate <- fp/(tn+fp)
  balanced <- (tp.rate+(1-fp.rate))/2
  f1 <- 2 * tp/(2 * tp + fp + fn)
  matthews <- (tp * tn - fp * fn)/sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  brier <- mean((criterion - predicted)^2)
  absloss <- mean(abs(criterion - predicted))
  logloss <- MLmetrics::LogLoss(predicted, criterion)

  aupr <- PRROC::pr.curve(scores.class0 = predicted, weights.class0 = criterion)$auc.integral

  
  if(is.na(matthews))
    matthews <- 0
  performance <- c(auc,accuracy,tp,fp,tn,fn,tp.rate,fp.rate,balanced,f1,matthews, brier, logloss, absloss, aupr)
  names(performance) <- c("auc","accuracy","tp","fp","tn","fn","tp.rate","fp.rate","balanced","f1","matthews", "brier", "logloss", "absloss", "aupr")
  return(performance)
}



calcAUC <- function(x,y){
  # calculate the area under the curve
  return(pROC::roc(y, x, direction = "<", quiet = T)$auc)
  
}


minmax <- function(x)
  return(c(min(x, na.rm = T), max(x, na.rm = T)))

moving_avg <- function(x,n=5){stats::filter(x,rep(1/n,n), sides=2)}



rocPoint <- function(criterion,predicted, threshold){
  # This function computes the true posisitve, false postive rate, and precision.
  # The reuslting numbers can be used to plot ROC and Precision-Recall curves
  
  # :param numeric vector criterion: observed class labels
  # :param numeric vector predicted: predicted probability of the positive class
  # :param double threshold: values on the "predicted vector" are are assigned to 
  #   the positive class if their value is greater than the threshold 
  
  
  predicted[is.na(predicted)] <- runif(sum(is.na(predicted)))
  predicted.t <- ifelse(predicted > threshold,1,0)
  
  tp <- as.numeric(sum(predicted.t==1&criterion==1))
  fp <- as.numeric(sum(predicted.t==1&criterion==0))
  tn <- as.numeric(sum(predicted.t==0&criterion==0))
  fn <- as.numeric(sum(predicted.t==0&criterion==1))
  tp_rate <- tp/(tp+fn)
  fp_rate <- fp/(tn+fp)
  precision <- tp/(tp + fp)
  return(c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision))
}

clrobustse <- function(fit.model, clusterid) {
  # computes robust standard errors

  # https://stackoverflow.com/questions/33927766/logit-binomial-regression-with-clustered-standard-errors
  N.obs <- length(clusterid)            
  N.clust <- length(unique(clusterid))  
  dfc <- N.clust/(N.clust-1)                    
  vcv <- vcov(fit.model)
  estfn <- sandwich::estfun(fit.model)
  uj <- apply(estfn, 2, function(x) tapply(x, clusterid, sum))
  N.VCV <- N.obs * vcv
  ujuj.nobs  <- crossprod(uj)/N.obs
  vcovCL <- dfc*(1/N.obs * (N.VCV %*% ujuj.nobs %*% N.VCV))
  lmtest::coeftest(fit.model, vcov=vcovCL)
}

grep_multi <- function(constraints, corpus){
  # filters those elements from a vector that include all substrings specified as constraints
  # :param character vector constraints: substrings that are ALL required to be in te filtered corpus
  # :param chracter vector corpus: vector of strings that is filtered
  
  if(length(constraints) == 1)
    return(grep(constraints, corpus, value = T, fixed = T) )
  ncorpus <- length(corpus)
  ix <- sapply(constraints, function(x)  grep(x, corpus, value = F, fixed = T)) 
  ix <- sapply(ix, function(x) 1:ncorpus %in% x)  
  return(corpus[apply(ix, 1, all)])
}




winsorize <- function(x, p = .025){
  
  quants <- quantile(x, c(p, 1-p))
  x[x < quants[1]] = quants[1]
  x[x > quants[2]] = quants[2]
  return(x)

  }

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}
sigmoid_inv <- function(x){
  return(-log(1/x -1))
}

makeTransparent = function(cols, alpha=0.15) {
  # make a color transparent
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")
  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(cols)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  newColor[is.na(cols)] <- NA
  return(newColor)
}

resetPar <- function() {
  dev.new()
  op <- par(no.readonly = TRUE)
  dev.off()
  op
}


brier_test <- function(p1,p2, y){
  n <- length(p1)
  pi = mean(y)
  dif = (2/n) * sum((p1 - p2) * pi - (p1-p2)*y)
  var = (4/(n^2)) * sum((p1-p2)^2 * pi * (1-pi))
  z = dif/(var^.5)
  return(2*pnorm(-abs(z)))
}



