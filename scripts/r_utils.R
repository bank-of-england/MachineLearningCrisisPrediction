# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

# This script contains helper functions that are used when estimating machine learning models in R. 



getWeightsFromCost <- function(costs,prior){
  # given the costs of the classes and the proportion of the classes in the data (prior)
  # this fucntion calculates the weights of the observations
  weights <- costs/(prior * costs[1] + (1-prior) * costs[2])
  
  return(weights)
}

getPrior <- function(data.input) {
  # get proprtion in the positive class
  prior <- sum(data.input[,1] == 1) / nrow(data.input)
  return(prior)
}

createFolds <- function(rows.data,n.folds=10,...){
  # cerate folds for cross-validation
  if(rows.data < n.folds)
    n.folds <- rows.data
  object.allocation <- sample(rows.data)
  fold.ix <- rep(1:n.folds,length.out=rows.data)
  out.folds <- cbind(fold.ix,object.allocation)
  out.folds <- out.folds[order(out.folds[,2]),]
  return(out.folds)
}


buildC50pruned <- function(data.input,costs = c(.5,.5),...)
  # build pruned classification tree
  return(buildC50(data.input,global.pruning=T,costs = costs,...))


buildC50 <- function(data.input,global.pruning=F,costs = c(.5,.5),...){
  # build classification tree
  weights <- getWeightsFromCost(costs,getPrior(data.input))
  case.weights <- ifelse(data.input[,1] == 1,weights[1],weights[2])
  
  criterion <- as.factor(data.input[,1])
  cues <- data.input[,-1]
  
  model.output <- C50::C5.0(y = criterion,x = cues,trials = 1,
                            weights = case.weights,
                            control = C50::C5.0Control(noGlobalPruning=!global.pruning, minCases = 1))##Tree is pruned
  return(model.output)
}

predicting <- function(object,...)
  # generic predicting method
  UseMethod("predicting",object)

predicting.C5.0 <- function(object, test.data){
  # make predictions for the decision tree
  cues <- test.data[, -1]
  predicted <- predict(object, newdata=cues, type="prob",trials=1)[,2]
  
  return(predicted)
}


buildLogisticElasticNet <- function(data.input, costs = c(.5,.5),
                                    folds = NULL, optimizeMeasure = "deviance", ...){
  
  weights <- getWeightsFromCost(costs, getPrior(data.input))
  case.weights <- ifelse(data.input[,1] == 1,weights[1],weights[2])
  
  cues <- data.input[, -1]
  criterion <- as.factor(data.input[,1])
  alphaVector <- seq(0,1,by = .1)
  
  if(is.null(folds)){
    folds <- createFolds(nrow(data.input))
    folds <- folds[order(folds[,2]),]
    folds <- folds[,1]
  }
  
  resCV <- lapply(alphaVector,function(x) glmnet::cv.glmnet(y = criterion,
                                                            x = as.matrix(cues),
                                                            family="binomial",
                                                            type.measure = optimizeMeasure, 
                                                            alpha = x,
                                                            foldid = folds, 
                                                            weights = case.weights)) #alpha =1 is lasso, alpha = 0 is ridge
  minError <- sapply(resCV,function(x)min(x$cvm))
  model.output <- resCV[[which.min(minError)]]
  
  return(model.output)
}


predicting.cv.glmnet <- function(object, test.data, ...){
  # make predictions for a linear model trained with glmnet
  cues <- test.data[,-1, drop = F]
  criterion <- test.data[,1]
  predicted <- predict(object = object, newx = as.matrix(cues), s = "lambda.min", type = "response")[,1]
  
  return(predicted)
}