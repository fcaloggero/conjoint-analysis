library(bayesm)
library(dummies)
library(reshape2)
library(knitr)
library(ggplot2)

setwd("~/Desktop/") 

# Load the dataset
load("respondents.RData")
taskV3 <- read.csv("task-cbc.csv", sep="\t")
head(taskV3)

df_scenarios <- read.csv("extra-scenarios-v3.csv")
load("efCode.RData")


apply(resp.data.v3[4:39], 2, function(x){tabulate(na.omit(x))})
task.mat <- as.matrix(taskV3[, c("screen", "RAM", "processor", "price", "brand")])
X.mat <- efcode.attmat.f(task.mat) 

#First, get the vector of prices from taskV3 and center it on its mean:
pricevec=taskV3$price-mean(taskV3$price)

#Next, we'll get the columns from X.mat that represent brand. 
#They should be the last three columns (check to be sure of this):
X.brands=X.mat[,9:11]

#Next, we're going to multiply each column in X.brands by pricevec. 
#This isn't matrix multiplication. It's taking the inner product of each column and pricevec: 
X.BrandByPrice = X.brands*pricevec 

#Now we'll combine X.mat and X.BrandsByPrice to get the X matrix we'll use for choice
#modeling: 
X.matrix=cbind(X.mat,X.BrandByPrice) 

det(t(X.matrix)%*%X.matrix)

#Let's get these responses out into their own data frame:
ydata=resp.data.v3[,4:39]

#Check to see if you have all 36 response variables:
names(ydata) 

#Make sure you have no missing data: 
ydata=na.omit(ydata)
summary(ydata)

#Now, convert ydata to matrix         
ydata=as.matrix(ydata)
fakers <- ydata
fakers$flag <- rowMeans(fakers, na.rm = FALSE, dims = 1)
# 1404, 1562, 3425

zowner <- 1 * ( ! is.na(resp.data.v3$vList3) )
head(zowner)
#Here's how you can create lgtdata: 

lgtdata = NULL
for (i in 1:424) { lgtdata[[i]]=list( y=ydata[i,],X=X.matrix )}
length(lgtdata)
str(lgtdata)
length(lgtdata) 

lgtdata[[3]]

#############################################################
### Ch. 2: Fitting HB MNL Model
#############################################################

require(bayesm)
#############################################################
#LINE 1: You're going to specify 100,000 iterations, and that every 5th sample is kept:
mcmctest=list(R=100000, keep=5)
#LINE 2: Create the “Data” list rhierMnlDP() expects:
Data1=list(p=3,lgtdata=lgtdata)   # p is choice set size. 
#LINE 3: testrun 1. Have to run these 3 lines of code multiple times to get to simulation goal
testrun1=rhierMnlDP(Data=Data1,Mcmc=mcmctest)
names(testrun1)
#############################################################

#samples from marginal posterior distributions
# 1st dimension is how many people, 2nd dim is how many betas, 3rd dim is # of simulations.
#Don't want autocorrelation, so we only keep every 5th value
betadraw1=testrun1$betadraw
dim(betadraw1)
head(betadraw1)

#############################################################
### DETERMINE THE BURN-IN PERIOD
# 1st object: person 1, 2nd obect: beta 1 (corresponds to dummy variable 1, which corresponds to screen 7"),
# 3rd is blank, and that's the 20000 simulated values.

# Resp. 1
plot(1:length(betadraw1[1,1,]),betadraw1[1,1,])
x <- 1:length(betadraw1[1,1,])
y <- betadraw1[1,1,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[1,2,]),betadraw1[1,2,])
x <- 1:length(betadraw1[1,2,])
y <- betadraw1[1,2,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[1,3,]),betadraw1[1,3,])
x <- 1:length(betadraw1[1,3,])
y <- betadraw1[1,3,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[1,5,]),betadraw1[1,5,])
x <- 1:length(betadraw1[1,5,])
y <- betadraw1[1,5,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

# Resp. 2
plot(1:length(betadraw1[2,1,]),betadraw1[2,1,])
x <- 1:length(betadraw1[2,1,])
y <- betadraw1[2,1,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[2,2,]),betadraw1[2,2,])
x <- 1:length(betadraw1[2,2,])
y <- betadraw1[2,2,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[2,3,]),betadraw1[2,3,])
x <- 1:length(betadraw1[2,3,])
y <- betadraw1[2,3,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw1[2,5,]),betadraw1[2,5,])
x <- 1:length(betadraw1[2,5,])
y <- betadraw1[2,5,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)


#This plot shows the 20000 simulated draws. Looking for betas to be within a "band" and to stabilize. 
#"Burning period" up to 10001, after that it's converging
plot(density(betadraw1[1,1,10001:20000],width=2)) # these are the points we've decided to hold onto
abline(v=0)
mn <- mean(betadraw1[1,1,10001:20000])
sd <- sd(betadraw1[1,1,10001:20000])
plot(density(betadraw1[2,1,10001:20000],width=2)) # these are the points we've decided to hold onto
plot(density(betadraw1[3,1,10001:20000],width=2)) # these are the points we've decided to hold onto

# Price
plot(density(betadraw1[1,8,10001:20000],width=2)) # these are the points we've decided to hold onto
abline(v=0)
plot(density(betadraw1[2,8,10001:20000],width=2)) # these are the points we've decided to hold onto
plot(density(betadraw1[3,8,10001:20000],width=2)) # these are the points we've decided to hold onto

plot(betadraw1[,,10001:20000],width=2) # these are the points we've decided to hold onto
plot(density(betadraw1[3,8,10001:20000],width=2)) # these are the points we've decided to hold onto


summary(betadraw1[1,1,10001:20000]) # here the beta is positive, so they like the 7" screen
summary(betadraw1[1,2,10001:20000]) # here the beta is negative, so they don't like the 10" screen
betameansoverall <- apply(betadraw1[,,10001:20000],c(2),mean)
betameansoverall # will know which features people like, high level


### PLOT BETAMEANSOVERLL
plotting <- as.data.frame(betameansoverall)
str(plotting)

plotx <- c('Screen 7', 'Screen 10', 'RAM 16 Gb', 'RAM 32 Gb', 'Processor 2 GHz', 'Processor 2.5 GHz',
           '$299', '$399', 'Somesong', 'Pear', 'Gaggle', 'Brand2*Price', 'Brand3*Price', 'Brand4*Price')

plotx <- as.character(plotx)
#Then turn it back into an ordered factor
plotx <- factor(plotx, levels=unique(plotx))

plotting <- cbind(plotx, plotting)

p6 <- ggplot(plotting, aes(x = plotx, y = plotting$betameansoverall, size = plotting$betameansoverall, fill=plotting$betameansoverall)) +
  geom_point(shape = 21) +
  theme_bw() +
  theme(plot.title = element_text(size=24,face="bold"), axis.title=element_text(size=20,face="bold")) +
  ggtitle("Overall Beta Means") +
  labs(x = "Attribute", y = "beta means") +
  scale_fill_continuous(low = "red3", high = "green3") +
  scale_size(range = c(4, 4)) +
  labs(size = "", fill = "Preference") +
  geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
p6


# summarizing the betas by percentile
perc <- apply(betadraw1[,,10001:20000],2,quantile,probs=c(0.05,0.10,0.25,0.5 ,0.75,0.90,0.95))
perc

#############################################################
### Ch. 3: Fitting a HB MNL model with prior ownership as a covariate
#############################################################

table(testrun1$Istardraw)

#include zowner as a covariate
zownertest=matrix(scale(zowner,scale=FALSE),ncol=1)
Data2=list(p=3,lgtdata=lgtdata,Z=zownertest)
testrun2=rhierMnlDP(Data=Data2,Mcmc=mcmctest)

names(testrun2)
dim(testrun2$Deltadraw)

betadraw2=testrun2$betadraw
summary(betadraw2[1,1,10001:20000]) # here the beta is positive, so they like the 7" screen
summary(betadraw2[1,2,10001:20000]) # here the beta is negative, so they don't like the 10" screen
betameansextra <- apply(betadraw2[,,10001:20000],c(2),mean)
betameansextra # will know which features people like, high level

chisq.test(betameansoverall,betameansextra)
apply(testrun2$Deltadraw[10001:20000,],2,mean) 
apply(testrun2$Deltadraw[10001:20000,],2,quantile,probs=c(0.05,0.10,0.25,0.5 ,0.75,0.90,0.95))
betadraw2=testrun2$betadraw
dim(betadraw2)

betameansprior <- apply(testrun2$Deltadraw[10001:20000,],2,mean)
betameansprior # will know which features people like, high level
betameansoverall

# Resp. 1
plot(1:length(betadraw2[1,1,]),betadraw2[1,1,])
x <- 1:length(betadraw2[1,1,])
y <- betadraw2[1,1,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw2[1,2,]),betadraw2[1,2,])
x <- 1:length(betadraw2[1,2,])
y <- betadraw2[1,2,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

# Resp. 2
plot(1:length(betadraw2[2,1,]),betadraw2[2,1,])
x <- 1:length(betadraw2[2,1,])
y <- betadraw2[2,1,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)

plot(1:length(betadraw2[2,2,]),betadraw2[2,2,])
x <- 1:length(betadraw2[2,2,])
y <- betadraw2[2,2,]
smoothingSpline = smooth.spline(x, y, spar=0.5)
plot(x,y)
lines(smoothingSpline, col='red', lwd=5)



### PLOT BETAMEANSPRIOR
plotting <- as.data.frame(betameansextra)
str(plotting)

plotx <- c('Screen 7', 'Screen 10', 'RAM 16 Gb', 'RAM 32 Gb', 'Processor 2 GHz', 'Processor 2.5 GHz',
           '$299', '$399', 'Somesong', 'Pear', 'Gaggle', 'Brand2*Price', 'Brand3*Price', 'Brand4*Price')

plotx <- as.character(plotx)
#Then turn it back into an ordered factor
plotx <- factor(plotx, levels=unique(plotx))

plotting <- cbind(plotx, plotting)

p7 <- ggplot(plotting, aes(x = plotx, y = plotting$betameansextra, size = plotting$betameansextra, fill=plotting$betameansextra)) +
  geom_point(shape = 21) +
  theme_bw() +
  theme(plot.title = element_text(size=24,face="bold"), axis.title=element_text(size=20,face="bold")) +
  ggtitle("Prior Ownership Beta Means") +
  labs(x = "Attribute", y = "beta means") +
  scale_fill_continuous(low = "red3", high = "green3") +
  scale_size(range = c(4, 4)) +
  labs(size = "", fill = "Preference") +
  geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
p7

### PLOT PARTS WORTH
###LOGODDS
plotx <- c('Screen 5"', 'Screen 7"', 'Screen 10"', 'RAM 8 Gb', 'RAM 16 Gb', 'RAM 32 Gb', 'Proc. 1.5 GHz', 'Proc. 2 GHz', 'Proc. 2.5 GHz',
           '$199', '$299', '$399', 'STC', 'Somesong', 'Pear', 'Gaggle', 'Brand1*Price*', 'Brand2*Price', 'Brand3*Price', 'Brand4*Price')
ploty <- c(-0.2447,	-0.1641,	0.4088,	-0.6836,	0.1116,	0.5720,	-2.3037,	1.0261,	1.2776,	2.6290,	0.2895,
       -2.9185,	0.5084,	-0.1854,	0.0371,	-0.3601,	-0.0864,	0.0830,	0.0200,	-0.0167)

plotx <- as.character(plotx)
#Then turn it back into an ordered factor
plotx <- factor(plotx, levels=unique(plotx))

pworth <- data.frame(plotx, ploty)

parts.worth <- ggplot(pworth, aes(x = plotx, y = ploty, fill=ploty)) +
  geom_bar(stat="identity") +
  theme_bw() +
  theme(plot.title = element_text(size=24,face="bold"), axis.title=element_text(size=20,face="bold")) +
  ggtitle("Parts Worth") +
  labs(x = "Attribute", y = "") +
  scale_fill_continuous(low = "red3", high = "green3") +
  scale_size(range = c(4, 4)) +
  labs(size = "", fill = "Preference") +
  geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
parts.worth

###ABSVAL

plotx <- c('$399', '$199',  'Proc. 1.5 GHz', 'Proc. 2.5 GHz', 'Proc. 2 GHz', 'RAM 8 Gb',  'RAM 32 Gb',  'STC', 'Screen 10"',
           'Gaggle',  '$299', 'Screen 5"', 'Somesong', 'Screen 7"', 'RAM 16 Gb', 'Brand1*Price*', 'Brand2*Price', 'Pear', 'Brand3*Price', 'Brand4*Price')
ploty <- c(21.0,	18.9,	16.5,	9.2, 7.4,	4.9,	4.1,	3.7,	2.9,	2.6,	2.1,	1.8,	1.3, 1.2,	0.8,	0.6,	0.6,	0.3,	0.1,	0.1)

plotx <- as.character(plotx)
#Then turn it back into an ordered factor
plotx <- factor(plotx, levels=unique(plotx))

pworth <- data.frame(plotx, ploty)

parts.worth <- ggplot(pworth, aes(x = plotx, y = ploty, fill=ploty)) +
  geom_bar(stat="identity") +
  coord_flip() +
  scale_x_discrete(limits = rev(levels(pworth$plotx))) +
  theme_bw() +
  theme(plot.title = element_text(size=24,face="bold"), axis.title=element_text(size=20,face="bold")) +
  ggtitle("Attribute Importance") +
  labs(x = "Attribute", y = "%") +
  scale_fill_continuous(low = "orange3", high = "green3") +
  scale_size(range = c(4, 4)) +
  labs(size = "", fill = "Percentage") +
  geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
parts.worth

###ABSVAL - Intuitive

plotx <- c('$199',  'Proc. 2.5 GHz', 'Proc. 2 GHz', 'RAM 32 Gb',  'STC', 'Screen 10"',
           '$299', 'RAM 16 Gb', 'Brand2*Price', 
           'Pear', 'Brand3*Price', 'Brand4*Price', 'Brand1*Price*', 'Screen 7"', 'Somesong', 'Screen 5"', 'Gaggle',  'RAM 8 Gb',  'Proc. 1.5 GHz', '$399')
ploty <- c(18.9,	9.2, 7.4,	4.1,	3.7,	2.9,	2.1,	0.8,	0.6,	0.3,	0.1,	-0.1, -0.6, -1.2,	-1.3, -1.8, -2.6,	-4.9,	-16.5,	-21.0)

plotx <- as.character(plotx)
#Then turn it back into an ordered factor
plotx <- factor(plotx, levels=unique(plotx))

pworth <- data.frame(plotx, ploty)

parts.worth <- ggplot(pworth, aes(x = plotx, y = ploty, fill=ploty)) +
  geom_bar(stat="identity") +
  coord_flip() +
  scale_x_discrete(limits = rev(levels(pworth$plotx))) +
  theme_bw() +
  theme(plot.title = element_text(size=24,face="bold"), axis.title=element_text(size=20,face="bold")) +
  ggtitle("Attribute Importance") +
  labs(x = "Attribute", y = "%") +
  scale_fill_continuous(low = "red3", high = "green3") +
  scale_size(range = c(4, 4)) +
  labs(size = "", fill = "Percentage") +
  geom_hline(aes(yintercept=0), colour="#990000", linetype="dashed")
parts.worth

#############################################################
### Ch. 4: Make customer choice prediction using the individual 
###        respondent’s model & goodness of fit & validation
#############################################################
## Prediction for the 36 choice sets using individuals respondents model
# Actual
actuals <- apply(resp.data.v3[4:39], 2, function(x){tabulate(na.omit(x))})
actuals <- t(actuals)

m <- matrix(custchoice, nrow =36,  byrow=F)
m2 <- t(m)
apply(m2, 2, function(x){tabulate(na.omit(x))})

betavec=matrix(betameansoverall,ncol=1,byrow=TRUE)
xbeta=X.matrix%*%(betavec)
dim(xbeta)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)
expxbeta2=exp(xbeta2)
rsumvec=rowSums(expxbeta2)
pchoicemat=expxbeta2/rsumvec
pchoicemat

pchoicemat2 <- round(pchoicemat*424,digits=0)
pchoicemat2



#this is the mean of the simulated values
betameans <- apply(betadraw1[,,10001:20000],c(1,2),mean)
betameans[2,]
str(betameans)

# 10000 simulated values for every person. That's the mean of the simulated values.
dim(betameans)
dim(t(betameans))

# matrix multiplication of x and beta. Gives log odds ration for preferences
xbeta=X.matrix%*%t(betameans)
dim(xbeta)
dim(X.matrix)

#This takes the 1st 3 rows and puts them in 3 columns (this is the 424 people times the
# 36 choice sets)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)


#eachnumber gives the predicted probability they would choose the option of that choice set.
#the row is the person, the column is the choice set
expxbeta2=exp(xbeta2)
#predicted choice probabilities
rsumvec=rowSums(expxbeta2) #adding the sum of the 3 options
pchoicemat=expxbeta2/rsumvec
head(pchoicemat) # each row gives you the predicted probability of the choice set for that person
dim(pchoicemat)

custchoice <- max.col(pchoicemat) #picks the corresponding column that has max probability
str(custchoice)
head(custchoice)
table(custchoice)

ydatavec <- as.vector(t(ydata))
str(ydatavec)
table(custchoice,ydatavec) #confusion matrix of what we predicted
# in 3661 instances, the model picked option one, and that was the correct number
# in 430 instances the model picked two, but that was wrong

require("pROC")
# good for 2x2 confusion matrix
roctest <- roc(ydatavec, custchoice, plot=TRUE)
auc(roctest)

# use multiclass: more appropriate for 3x3
roctestMC <- multiclass.roc(ydatavec, custchoice, plot=TRUE)
auc(roctestMC)
###############
#### 2nd Model
#this is the mean of the simulated values
betameans <- apply(betadraw2[,,10001:20000],c(1,2),mean)
betameans[1,]
str(betameans)

# 10000 simulated values for every person. That's the mean of the simulated values.
dim(betameans)
dim(t(betameans))

# matrix multiplication of x and beta. Gives log odds ration for preferences
xbeta=X.matrix%*%t(betameans)
dim(xbeta)
dim(X.matrix)

#This takes the 1st 3 rows and puts them in 3 columns (this is the 424 people times the
# 36 choice sets)
xbeta2=matrix(xbeta,ncol=3,byrow=TRUE)
dim(xbeta2)


#eachnumber gives the predicted probability they would choose the option of that choice set.
#the row is the person, the column is the choice set
expxbeta2=exp(xbeta2)
#predicted choice probabilities
rsumvec=rowSums(expxbeta2) #adding the sum of the 3 options
pchoicemat=expxbeta2/rsumvec
head(pchoicemat) # each row gives you the predicted probability of the choice set for that person
dim(pchoicemat)

custchoice <- max.col(pchoicemat) #picks the corresponding column that has max probability
str(custchoice)
head(custchoice)

ydatavec <- as.vector(t(ydata))
str(ydatavec)
table(custchoice,ydatavec) #confusion matrix of what we predicted
# in 3661 instances, the model picked option one, and that was the correct number
# in 430 instances the model picked two, but that was wrong

require("pROC")
# good for 2x2 confusion matrix
roctest <- roc(ydatavec, custchoice, plot=TRUE)
auc(roctest)

# use multiclass: more appropriate for 3x3
roctestMC <- multiclass.roc(ydatavec, custchoice, plot=TRUE)
auc(roctestMC)
###################
# the smaller the better. Good for picking between models. Choose smaller loglikihood
logliketest1 <- testrun1$loglike
mean(logliketest)
hist(logliketest)

logliketest2 <- testrun2$loglike
mean(logliketest)
hist(logliketest)

#Now you can use Custchoice to predict the choices for the 36 choice sets.
m <- matrix(custchoice, nrow =36,  byrow=F)
m2 <- t(m)
apply(m2, 2, function(x){tabulate(na.omit(x))})

#############################################################
### Ch. 5: Predicting extra scenarios, as well as the 36 choice sets, 
###        using betas from all the pooled respondents
#############################################################
ex_scen <- read.csv("extra-scenarios.csv")
Xextra.matrix <- as.matrix(ex_scen[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9",
                                      "V10","V11","V12","V13","V14")])


# predict extra scenarios based on individual models
xextrabetaind=Xextra.matrix %*% (t(betameans))
xbetaextra2ind=matrix(xextrabetaind, ncol=3, byrow=TRUE)
dim(xbetaextra2ind)

expxbetaextra2ind <- exp(xbetaextra2ind)
rsumvecind <- rowSums(expxbetaextra2ind)
pchoicematind <- expxbetaextra2ind/rsumvecind
dim(pchoicematind)
head(pchoicematind)

custchoiceind <- max.col(pchoicematind)
head(custchoiceind)

extra1 <- custchoiceind[1:424]
extra2 <- custchoiceind[425:848]
table(extra1)
table(extra2)


