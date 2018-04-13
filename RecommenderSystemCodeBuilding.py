import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pylab as graphplot
from collections import Counter
from sklearn.metrics import jaccard_similarity_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle


followerFile = pd.read_csv("follows.csv") 
interestFile = pd.read_csv("interests.csv")
FollowerFileWithTargetZero = pd.read_csv("TargetZeroList.csv") 

def ComputeJaccardSimilarityIndexes(listToAnalyze):
	#Create two new lists to store the jaccard similarity scores
	JaccardFolloweeList = list()
	JaccardFollowerList= list()
	JaccardInterestList= list()

	for UserNumber in range(0,len(listToAnalyze)):
			
		if listToAnalyze[UserNumber,0] in FolloweeList.keys() and listToAnalyze[UserNumber,1] in FolloweeList.keys():
			JaccardFolloweeValue = float(len(list(set(FolloweeList[listToAnalyze[UserNumber,0]]).intersection(FolloweeList[listToAnalyze[UserNumber,1]]))))/(len(set(FolloweeList[listToAnalyze[UserNumber,0]] + FolloweeList[listToAnalyze[UserNumber,1]])))	
	
		else:
			JaccardFolloweeValue = 0


		if listToAnalyze[UserNumber,1] in FollowerList.keys() and listToAnalyze[UserNumber,0] in FollowerList.keys():
			JaccardFollowerValue = float(len(list(set(FollowerList[listToAnalyze[UserNumber,0]]).intersection(FollowerList[listToAnalyze[UserNumber,1]]))))/(len(set(FollowerList[listToAnalyze[UserNumber,0]] + FollowerList[listToAnalyze[UserNumber,1]])))
	
		else:
			JaccardFollowerValue = 0

		#For interest we have to find that the users ie follower as well followee have to exist in the interest user file or not
		if (listToAnalyze[UserNumber,0] in InterestUserList.keys()) and (listToAnalyze[UserNumber,1] in InterestUserList.keys()):
			JaccardInterestValue = float(len(list(set(InterestUserList[listToAnalyze[UserNumber,0]]).intersection(InterestUserList[listToAnalyze[UserNumber,1]]))))/(len(set(InterestUserList[listToAnalyze[UserNumber,0]] + InterestUserList[listToAnalyze[UserNumber,1]])))
	
		else:
			JaccardInterestValue = 0
		

		#Append FolloweeIndex for the pair to Followee list
		JaccardFolloweeList.append(JaccardFolloweeValue)

		#Append FolloweeIndex to Follower list
		JaccardFollowerList.append(JaccardFollowerValue)

		#Append Interest Jaccard score to Interest list
		JaccardInterestList.append(JaccardInterestValue)
	
	return JaccardFolloweeList,JaccardFollowerList,JaccardInterestList

#Populate the training and testing datasets
def populateDatasets():
	
	#create lists which will be populated
	TrainTargetList = list()
	TestTargetList = list()
	TrainList = list()
	TestList = list()

	DataframeTrainingInputMerged = [followerFile.iloc[:40000,2:],FollowerFileWithTargetZero.iloc[:40000,2:]]
	DataframeTrainingInputMerged = pd.concat(DataframeTrainingInputMerged)

	#Shuffle the dataset created to haveshuffled target values in the dataset
	DataframeTrainingInputMerged = shuffle(DataframeTrainingInputMerged)

	#First three columns are the training jaccard inputs
	#	Format of dataset Jaccard1 Jaccard2 Jaccard3 Target
	TrainList = DataframeTrainingInputMerged.iloc[:,:3]

	#Last column i.e. column 3 is the target column
	TrainTargetList = DataframeTrainingInputMerged.iloc[:,3]


	#Join the two dataset i.e. target =1 and target = 0 datasets to one dataframe
	DataframeTestingInputMerged = [followerFile.iloc[40001:,2:],FollowerFileWithTargetZero.iloc[40001:,2:]]
	DataframeTestingInputMerged = pd.concat(DataframeTestingInputMerged)

	#Shuffle the dataset
	DataframeTestingInputMerged = shuffle(DataframeTestingInputMerged)

	#First three columns are the testing jaccard inputs
	#	Format of dataset Jaccard1 Jaccard2 Jaccard3 Target
	TestList = DataframeTestingInputMerged.iloc[:,:3]

	#Last column i.e. column 3 is the target column
	TestTargetList = DataframeTestingInputMerged.iloc[:,3]

	#return the lists created
	return TrainTargetList,TestTargetList,TrainList,TestList


#This function takes the false positive rate,true positive rate and the roc_area under the curve calculated before. It also takes the name of the model as an argument
def plotRocCurve(fpr, tpr,roc_auc,NameOfModel):
	lw = 1
	plt.plot(fpr, tpr,
		     lw=lw, label='ROC curve for ' + NameOfModel + ' ' + str(roc_auc))
	plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example for: ' + NameOfModel)
	plt.legend(loc="lower right")
	plt.show()
##############################################################################################################################################################

#Who does the user follows.
#creates a dictionary of user id as the key and the users they follow as values
FollowerList = {k:interest["followee_id"].tolist() for k,interest in followerFile.groupby("follower_id")}

#Who follows the particular users
#creates a dictionary of user id as the key and the users who follow the user as values
FolloweeList = {k:interest["follower_id"].tolist() for k,interest in followerFile.groupby("followee_id")}

#Which categories the user likes
#Creates a list of user id as key and the interests they have as values
InterestUserList = {k:interest["category"].tolist() for k,interest in interestFile.groupby("user_id")} 


listOfUsersTargetOne = list()	#List of users with target value 1 i.e. follower follows followee
listOfUsersTargetZero = list()	#List of users with target value 0 i.e. follower doesnot follows followee

JaccardFolloweeListTargetOne = list()	#List to store jaccard index of similarity between the users that follow the client
JaccardFollowerListTargetOne = list()	#List to store jaccard index of similarity between the users that the client follows
JaccardInterestListTargetOne = list()   #List to store jaccard index of interests between the users

JaccardFolloweeListTargetZero = list()	#List to store jaccard index of similarity for target zero file
JaccardFollowerListTargetZero = list()	#List to store jaccard index of similarity for target zero file
JaccardInterestListTargetZero = list()   #List to store jaccard index of interests between the users for target zero file


#1.Create a list of users and their followee from the folows.csv
#2. Now we have the list for which the target is zero since we know that these users already follow the other users
#3. We calculate the jaccard index for these user combinations i.e. follower followee combination

for rowIndex in range(0,len(followerFile)):
	listOfUsersTargetOne.append(followerFile.iloc[rowIndex,:])

#Populate numpy array for target zero lists
for rowIndex in range(0,len(FollowerFileWithTargetZero)):
	listOfUsersTargetZero.append(FollowerFileWithTargetZero.iloc[rowIndex,:])

listOfUsersTargetOne = np.array(listOfUsersTargetOne)
listOfUsersTargetZero = np.array(listOfUsersTargetZero)

#Call the function to find jaccard Similarity index for Target One list
JaccardFolloweeValueTargetOne,JaccardFollowerValueTargetOne,JaccardInterestListTargetOne = ComputeJaccardSimilarityIndexes(listOfUsersTargetOne)

#Compute Jaccard INdex for list with no connections in between users
JaccardFolloweeValueTargetZero,JaccardFollowerValueTargetZero,JaccardInterestListTargetZero = ComputeJaccardSimilarityIndexes(listOfUsersTargetZero)	


#Populate Dataframe for target 1 elements
followerFile['JaccardFollowerIndex'] = JaccardFollowerValueTargetOne
followerFile['JaccardFolloweeIndex'] = JaccardFolloweeValueTargetOne
followerFile['JaccardInterestIndex'] = JaccardInterestListTargetOne
targetListForPositiveCases = [1] * len(followerFile)	#target = 1 since they follow the followee
followerFile['Target'] = targetListForPositiveCases		#Create a target column in the dataframe

#Populate dataframe with target 0 elements
FollowerFileWithTargetZero['JaccardFollowerIndex'] = JaccardFollowerValueTargetZero
FollowerFileWithTargetZero['JaccardFolloweeIndex'] = JaccardFolloweeValueTargetZero
FollowerFileWithTargetZero['JaccardInterestIndex'] = JaccardInterestListTargetZero
targetListForNoConnectionCases = [0] * len(FollowerFileWithTargetZero) #target = 0 since they don't follow the followee as per the dataset i.e. no info given
FollowerFileWithTargetZero['Target'] = targetListForNoConnectionCases	#Create a target column in the dataframe


########################################################################################################################################################
#Creating datasets for analysis
TrainTargetList = list()
TestTargetList = list()

TrainList = list()
TestList = list()

#Populate the training and testing lists using the below function

TrainTargetList,TestTargetList,TrainList,TestList = populateDatasets()

########################################################################################################################################################
#Model Training below this comment

#Train a logistic regression model on the input dataset

LogisticRegression = linear_model.LogisticRegression()

LogisticRegression.fit (TrainList,TrainTargetList)
joblib.dump(LogisticRegression,'LogisticRegression.pkl')
print 'Accuracy for Logistic Regression is :' ,LogisticRegression.score(TestList,TestTargetList)

######################
#Plot roc for logistic
fpr = dict()
tpr = dict()
roc_auc = dict()
prob_y = LogisticRegression.predict_proba(TestList)
fpr, tpr,thresholds = roc_curve(TestTargetList, prob_y[:,1]) #the first column has the scores for target 1
roc_auc = auc(fpr, tpr)

plotRocCurve(fpr, tpr,roc_auc,'Logistic Regression')
######################

######################
#Train a neural network model on the input dataset
NeuralModel = MLPClassifier()

NeuralModel.fit(TrainList,TrainTargetList)
joblib.dump(NeuralModel,'NeuralModel.pkl')
print 'Accuracy for Neural Network is :' ,NeuralModel.score(TestList,TestTargetList) 

######################
#Plot roc for neural network
fpr = dict()
tpr = dict()
roc_auc = dict()
prob_y = NeuralModel.predict_proba(TestList)
fpr, tpr,thresholds = roc_curve(TestTargetList, prob_y[:,1]) #the first column has the scores for target 1
roc_auc = auc(fpr, tpr)

plotRocCurve(fpr, tpr,roc_auc,'Neural Network')
######################




