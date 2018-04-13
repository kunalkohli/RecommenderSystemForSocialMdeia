from sklearn.externals import joblib
import pandas as pd
import numpy as np


followerFile = pd.read_csv("follows.csv") 
interestFile = pd.read_csv("interests.csv")
FollowerFileWithTargetZero = pd.read_csv("TargetZeroList.csv") 

FollowerList = {k:interest["followee_id"].tolist() for k,interest in followerFile.groupby("follower_id")}

#Who follows the particular users
#creates a dictionary of user id as the key and the users who follow the user as values
FolloweeList = {k:interest["follower_id"].tolist() for k,interest in followerFile.groupby("followee_id")}

#Which categories the user likes
#Creates a list of user id as key and the interests they have as values
InterestUserList = {k:interest["category"].tolist() for k,interest in interestFile.groupby("user_id")} 


#Compute the jaccard Indexes of the user combination ( user input by client) and all the users in the system for which we have to make a recommendation
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



UserInput = raw_input("enter the user Id for which you want the recommendation:")

try:
	UserInput = int(UserInput)
except ValueError:
	# UserInput was not an integer, hence exit the function
	exit()


#Create fresh lists for recommending the user suggestions
JaccardFolloweeList = list()
JaccardFollowerList = list()
JaccardInterestList = list()

ListOfAllUsers = list()
ListOfAllUsers.append(followerFile.iloc[:,0].values)
ListOfAllUsers.append(followerFile.iloc[:,1].values)
ListOfAllUsers.append(interestFile.iloc[:,0].values)
ListOfAllUsers = [item for sublist in ListOfAllUsers for item in sublist]
ListOfAllUsers = set(ListOfAllUsers)
ListOfAllUsers = list(ListOfAllUsers)


#create a list of user recommendations
listOfRecommendedUsers = list()

#Load the model from the trained model
classifier = joblib.load('NeuralModel.pkl')

#Create a list storing the probabilites of recommendation for every user combination.
UserPredictionList = list()

if UserInput in ListOfAllUsers:
	#create a list to store jaccard similarity indexes for every user combination
	jaccardSimilarityList = list()
	for userId in ListOfAllUsers:
		#To not append the user id as [x,x] in the system i.e. to find recom for 4 we would not want to test combination [4,4]
		if userId != UserInput:
			jaccardSimilarityList.append([UserInput,userId])
			numpyjaccardSimilarityList = np.array(jaccardSimilarityList)

	JaccardFolloweeList,JaccardFollowerList,JaccardInterestList = ComputeJaccardSimilarityIndexes(numpyjaccardSimilarityList)

	for index in range(0,len(JaccardFolloweeList)):
		listToPredict= list()
		listToPredict.append([JaccardFolloweeList[index],JaccardFollowerList[index],JaccardInterestList[index]])
		listToPredict = np.array(listToPredict)
		
		#Predict probabilites for accepting the connection request
		probabilities = classifier.predict_proba(listToPredict)[:,1]
		UserPredictionList.append([ListOfAllUsers[index],probabilities[0]])	
	
	UserPredictionList = np.array(UserPredictionList)
	UserPredictionList = UserPredictionList[UserPredictionList[:,1].argsort()[::-1]]

	print 'Top 10 users of recommended Users for user ID : ' + str(UserInput)  + ' are :'
	print UserPredictionList[:10,0]	

else:
	print 'Please enter a valid user Id for the recommender System. User ID : ' + str(UserInput) + ' is not available in the system.'
	exit()



