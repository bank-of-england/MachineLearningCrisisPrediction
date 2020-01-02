# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 
# This script installs all R packages required to reproduce the results.
install.packages("devtools")
# packages and their versions used
packages <- list(
  c("dplyr", "0.8.3"),
  c("C50", "0.1.2"),
  c("MLmetrics", "1.1.1"),
  c("openxlsx", "4.1.0.1"),
  c("pROC", "1.15.3"),
  c("plotrix", "3.7-6"),
  c("lmtest", "0.9-37"),
  c("car", "1.4-5"),
  c("sandwich", "1.5-1"),
  c("pROC", "1.15.3"),
  c("PRROC", "1.3.1")
)


lapply(packages, function(package) install.packages(package[1],
                                                    version = package[2],
                                                    repos = "http://cran.us.r-project.org"
                                                    ))
