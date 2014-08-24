MachineLearningProject
======================

R code and markdown for Practical Machine Learning Coursera module
------------------------------------------------------------------
This project is submitted to address the requirements of the above module, as
per the assignment 2 project briefing.

The essential steps in the included R script are as follows :
1. Check for the presence of the required files in the defined data directory
(line 12 of code) and, if absent, download it.
2. The test data is examined only the first 2 lines to obtain the list of parameters
(column names) which can be used as inputs to prediction (i.e. there is no use
fitting a machine learning model that receives the value NA). The test data is
deleted from R memory for all purposes of training and validation.
3. The training data is internally pre-processed and cross-validated as 10-fold
partitions.
4.







From the observations it can be seen that some of the available values are highly
skewed with only one or two extreme values




Citation
========
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity
Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference
in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz39tstDjrm


