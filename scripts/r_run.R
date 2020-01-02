# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

# This script is called from Python and estimates machine learning models in R. 
# required packages: C50, glmnet

verbose = F
myArgs <- commandArgs(trailingOnly = TRUE)
# costs <- c(as.numeric(myArgs[[1]]), as.numeric(myArgs[[2]])) # costs can also be passed from Python
# but we define them below

ident = myArgs[[3]]
n_kernels = as.numeric(myArgs[[4]])
heuristic_complexity <- as.numeric(myArgs[[5]])
algoName = simplify2array(myArgs[[6]])

source("scripts/r_utils.R")
setwd("r_data/")

nameTrain <- paste0("train_in",ident,".csv")
nameTest <- paste0("test_in",ident,".csv")
nameFold <- paste0("cv_fold_vector",ident,".csv")

# read data from the hard drive. It will be deleted afterwards from Python
train <- read.csv(nameTrain, sep = "\t")[,-1]
test <- read.csv(nameTest, sep = "\t")[,-1]
folds = read.csv(nameFold, sep = "\t", header = F)[,-1] + 1
costs <- c(1- mean(train[,1]), mean(train[,1])) # we weigh objects such that training andtest set contribute equally to the training set

mean(train[,1])
nc <- ncol(train)-1

out <- rep(NA, nrow(test))


if(verbose){
  print("Data read")
  print(paste0("Costs: ",costs))
}

if(algoName == "r_c50"){ # Train decision tree
  model <- buildC50pruned(train, costs = costs)
  out <-  predicting(model,test)
}



write.csv(out,file = paste0("out_to_python_", algoName, ident,".csv"))


