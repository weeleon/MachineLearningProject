#project.R is a machine learning predictive script that classifies a set
#of wearable device parameters as one of five classes A,B,C,D or E.
#
#

#required packages
library(caret)
library(gbm)
library(plyr)
library(rpart)
library(randomForest)

#initialise path to data location
dataDirectory <- "/Users/lenw/Documents/Coursera_Offline/Data_Science/Machine_Learning_Project"
setwd(dataDirectory)

#prepare to read training and testing data
testData <- "pml-testing.csv" #relative path to testing data
trainData <- "pml-training.csv" #relative path to training data

#if the test and training machine learning dataset does not exist in dataDirectory
#then download it
if (!file.exists(trainData)) {
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",trainData,method="curl")      
      dateDownloaded <- paste(date(),"CEST (Copenhagen, Denmark)")
      write.table(dateDownloaded,file="pml-training-download-date.txt",col.names=F,row.names=F,quote=F)
      }

if (!file.exists(testData)) {
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",testData,method="curl")      
      dateDownloaded <- paste(date(),"CEST (Copenhagen, Denmark)")
      write.table(dateDownloaded,file="pml-testing-download-date.txt",col.names=F,row.names=F,quote=F)
      }



#load the training data set
indata <- read.csv(trainData)
#examine the distribution of data in the file before data splitting :
TrainingSummaryTable <- with(indata, table(user_name,classe))
#print the training data distribution on the console
TrainingSummaryTable



#only read the test data set to get the names of the usable columns for prediction
testing <- read.csv(testData,nrows=2)

#the final testing and evaluation can only be done on 60 non-NA columns
#coumns with NA values can be used to predict "classe" in the final set
#therefore there is no reason to build these into a prediction model
usefulcolumns <- !is.na(testing[1,]) #what are the usable prediction parameters?
indata <- indata[,usefulcolumns] #only select these as prediction parameters

#likewise, we do not want to allow machine learning to construct a prediction
#from time stamps, or user name, or window numbers, so drop these as well
indata <- indata[,c(8:60)] #but exclude the identifier data in the first seven columns

#remove the testing data so there is no way to accidentally visualise it
rm(testing)
rm(usefulcolumns)



#check data for zero variance parameters
nzv <- nearZeroVar(indata[,-53], saveMetrics = TRUE)
if (sum(nzv$nzv) > 0) { print("Warning - there are parameters with zero or near-zero variance.")}
rm(nzv)
#- a zero sum implies that none of the 52 parameters are zero or near-zero variance

#NB : There are no zero-variance parameters among the available 52 columns


#check for tightly correlated parameters which would be redundant for prediction
#we arbitrarily set a correlation absolute threshold of 0.90
#filter and name the correlated variables
highlyCor <- findCorrelation(cor(indata[,-53]), cutoff = 0.90)
names(indata)[highlyCor]
#remove variables that are highly correlated with other variables
filteredInData <- indata[,-highlyCor]

#NB : This pre-processing step has removed 7 variables from the column list
#resulting in 45 parameters available for machine-based prediction



#generate a generic command sequence for training control for any learning
fitControl <- trainControl(
      ## 10-fold repeated cross-validation
      method = "repeatedcv", number = 10,
      ## repeated five times
      repeats = 5
      ## no pre-processing scaling and centering not critical for tree methods
      )



# ----------- create calls to the training object

## learning object 1 : simple classification tree with all variables
set.seed(33524) #set initial seed
treeFit1 <- train(classe ~ ., data = indata, method = "rpart",
                  trControl = fitControl,
                  tuneGrid = expand.grid(.cp=c(0.0001,0.001,0.003,0.01,0.03,0.1))
                  )

## learning object 2 : simple classification tree with correlation-filtered variables
set.seed(33524) #set initial seed
treeFit2 <- train(classe ~ ., data = filteredInData, method = "rpart",
                  trControl = fitControl,
                  tuneGrid = expand.grid(.cp=c(0.0001,0.001,0.003,0.01,0.03,0.1))
)
## this simple object sets out the baseline accuracy at roughly 83% for a
## complexity parameter of 0.003



## learning object 3 : random forest classification with all variables
set.seed(33524) #reset initial seed
forestFit1 <- train(classe ~ ., data = indata, method = "rf",
                    trControl = fitControl,
                    allowParallel = TRUE,
                    tuneGrid = expand.grid(.mtry=c(23,27,45,52))
                    )


## learning object 4 : random forest classification with correlation-filtered variables
set.seed(33524) #reset initial seed
forestFit2 <- train(classe ~ ., data = filteredInData, method = "rf",
                    trControl = fitControl,
                    allowParallel = TRUE,
                    tuneGrid = expand.grid(.mtry=c(23,27,45))
                    )

## for parallel processing ...
library(doMC)
registerDoMC(cores=2)
fitControl <- trainControl(
      ## 10-fold repeated cross-validation
      method = "repeatedcv", number = 10,
      ## repeated five times
      repeats = 5,
      ## no pre-processing scaling and centering not critical for tree methods
      allowParallel = TRUE
)

## learning object 5 : gradient boosting with all variables
set.seed(33524) #set initial seed
gbmFit1 <- train(classe ~ ., data = indata, method = "gbm",
                 trControl = fitControl,
                 ##switch off verbose
                 verbose = FALSE
                 )

## learning object 6 : gradient boosting with correlation-filtered variables
set.seed(33524) #set initial seed
gbmFit2 <- train(classe ~ ., data = filteredInData, method = "gbm",
                 trControl = fitControl,
                 ##switch off verbose
                 verbose = FALSE
                 )



# ----------- apply the best model with the best tuning parameter
# to the whole training set
testing <- read.csv(testData)
usefulcolumns <- !is.na(testing[1,]) #what are the usable prediction parameters?
finalTest <- testing[,usefulcolumns] #only select these as prediction parameters
finalTest <- finalTest[,c(8:60)] #but exclude the identifier data in the first seven columns

treeFit1
## maximal accuracy 94% with 10-fold cross-validation in 5 reps

treeFit2
## maximal accuracy 93.5% with 10-fold cross-validation in 5 reps
## but faster run on reduced parameter set (removed close correlations)

forestFit1
## maximal accuracy 99.6% with 10-fold cross-validation in 5 reps

forestFit2
## maximal accuracy 99.5% with 10-fold cross-validation in 5 reps
## but faster run on reduced parameter set (removed close correlations)

## choose forestFit2 for final prediction
answer <- predict(forestFit2, newdata=finalTest)
answer
## submission of this answer yields 20/20 correct on the final test set


## prepare to write answer file
pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
}

## export answer files
pml_write_files(answer)


#visualisation and plotting for the purpose of the report
id <- rep("CART-52",6)
treeComp1 <- cbind(treeFit1$results,id)
id <- rep("CART-45",6)
treeComp2 <- cbind(treeFit2$results,id)
treeAccuracy <- rbind(treeComp1,treeComp2)
treeAccuracy
## --------------------------
id <- rep("CART-52",4)
forestComp1 <- cbind(forestFit1$results,id)
id <- rep("CART-45",3)
forestComp2 <- cbind(forestFit2$results,id)
forestAccuracy <- rbind(forestComp1,forestComp2)
forestAccuracy


lp <- ggplot(data=treeAccuracy,aes(x=cp,y=Accuracy,group=id,colour=id))
lp <- lp + geom_line()+geom_point(size=4)
lps <- lp + scale_colour_discrete(name="Training Data Set",breaks=c("CART-52", "CART-45"),labels=c("Pre-processed", "Correlation-Filtered")) + scale_shape_discrete(name="Training Data Set",breaks=c("CART-52", "CART-45"),labels=c("Pre-processed", "Correlation-Filtered")) +ylim(0,1) +xlab("Complexity Parameter, Cp") +ylab("Accuracy (repeated cross-validation)") +ggtitle("Fig.1. In-sample Accuracy of\nCART Decision Tree by Tuning Parameter")
lps
dev.copy(png,"fig1.png",height=480,width=600,units="px"); dev.off();
## --------------------------
lp <- ggplot(data=forestAccuracy,aes(x=mtry,y=Accuracy,group=id,colour=id))
lp <- lp + geom_line()+geom_point(size=4)
lps <- lp + scale_colour_discrete(name="Training Data Set",breaks=c("CART-52", "CART-45"),labels=c("Pre-processed", "Correlation-Filtered")) + scale_shape_discrete(name="Training Data Set",breaks=c("CART-52", "CART-45"),labels=c("Pre-processed", "Correlation-Filtered")) +ylim(0,1) +xlab("Number of Tree Parameters, mtry") +ylab("Accuracy (repeated cross-validation)") +ggtitle("Fig.2. In-sample Accuracy\nof Random Forest by Tuning Parameter")
lps
dev.copy(png,"fig2.png",height=480,width=600,units="px"); dev.off();



