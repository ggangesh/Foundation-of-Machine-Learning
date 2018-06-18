import numpy as np
from copy import deepcopy
import pandas as pd
import pickle

# USED MACROS

NeauralNetwork = [15, 30, 30, 1]
iterations = 3000
learnrate  = 0.0002

#features are mapped with weights
workclassd = {'Private': 3, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':2, 'Local-gov':1, 'State-gov':1, 'Without-pay':0, 'Never-worked':0,'?':0}
educationd = {'Bachelors':8, 'Some-college':7, '11th':5, 'HS-grad':4, 'Prof-school':9, 'Assoc-acdm':10, 'Assoc-voc':9, '9th':3, '7th-8th':2, '12th':6, 'Masters':11, '1st-4th':0, '10th':4, 'Doctorate':12, '5th-6th':1, 'Preschool':0,'?':0}
maritalstatusd = { 'Married-civ-spouse':5, 'Divorced':3, 'Never-married':2, 'Separated':1, 'Widowed':4, 'Married-spouse-absent':0, 'Married-AF-spouse':0,'?':0}
occupationd =  {'Tech-support':1, 'Craft-repair':0, 'Other-service':0, 'Sales':3, 'Exec-managerial':5, 'Prof-specialty':4, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':1,'?':0}
relationshipd = {'Wife':2, 'Own-child':0, 'Husband':3, 'Not-in-family':0, 'Other-relative':0, 'Unmarried':1,'?':0}
raced = {'White':2, 'Asian-Pac-Islander':0, 'Amer-Indian-Eskimo':0,'Other':0, 'Black':1,'?':0}
sexd = {'Female':1, 'Male':2,'?':0}
countryd = {'United-States':5, 'Cambodia':0, 'England':3, 'Puerto-Rico':0, 'Canada':4, 'Germany':4, 'Outlying-US(Guam-USVI-etc)':0, 'India':1, 'Japan':3, 'Greece':2, 'South':0, 'China':2, 'Cuba':0, 'Iran':2, 'Honduras':0, 'Philippines':0, 'Italy':3, 'Poland':3, 'Jamaica':1, 'Vietnam':2, 'Mexico':1, 'Portugal':3, 'Ireland':3, 'France':3, 'Dominican-Republic':1, 'Laos':0, 'Ecuador':1, 'Taiwan':2, 'Haiti':0, 'Columbia':1, 'Hungary':1, 'Guatemala':0, 'Nicaragua':0, 'Scotland':2, 'Thailand':2, 'Yugoslavia':2, 'El-Salvador':1, 'Trinadad&Tobago':3, 'Peru':1, 'Hong':1, 'Holand-Netherlands':4,'?':0}

def sigm(x):
	return 1/(1+np.exp(-x))

def dsigm(x):
	return x*(1-x)


def trainnet(xdata, ydata):
	wts = []
	for i in range(3):
		wts.append(np.zeros((NeauralNetwork[i], NeauralNetwork[i+1]), dtype=np.float64))
	
	t1 = 1.0
	t2 = 0.0

	for i in range(iterations):

		# feedforward
		l1 = sigm(np.dot(xdata, wts[0])) # input 
		l2 = sigm(np.dot(l1, wts[1]))	# hidden
		l3 = sigm(np.dot(l2, wts[2]))	# hidden

		#backpropogation
		step3 = (ydata - l3) * dsigm(l3)
		wts[2] += learnrate * l2.T.dot(step3)

		step2 = (step3.dot(wts[2].T)) * dsigm(l2)
		wts[1] += learnrate * l1.T.dot(step2)
		
		step1 = step2.dot(wts[1].T) * dsigm(l1)
		wts[0] += learnrate * xdata.T.dot(step1)
		t2 = t1
		t1 = (np.mean(np.abs(ydata - l3)))
		if((i % 50) == 0):
			print (i, '			', t1 ,' 	', (t1 - t2))
		#if(t2 < t1):
		#	break
	return wts
#End of function train


print  ('reading the input and generating features ...')

inputdata = pd.read_csv('./train.csv', delimiter=",")
xdata = (inputdata.drop(inputdata.columns[-1], axis=1)).values
ydata = np.array((inputdata.drop(inputdata.columns[0:-1], axis=1)).values, dtype=np.int64)

#features 
for i in range(len(xdata)):
	xdata[i][2] = workclassd[xdata[i][2].lstrip()]
	xdata[i][4] = educationd[xdata[i][4].lstrip()]
	xdata[i][6] = maritalstatusd[xdata[i][6].lstrip()]
	xdata[i][7] = occupationd[xdata[i][7].lstrip()]
	xdata[i][8] = relationshipd[xdata[i][8].lstrip()]
	xdata[i][9] = raced[xdata[i][9].lstrip()]
	xdata[i][10] = sexd[xdata[i][10].lstrip()]
	xdata[i][14] = countryd[xdata[i][14].lstrip()]

#normalize the data
xdata = np.array(xdata, dtype=np.int64)
xdata = (xdata - xdata.min(0))/xdata.ptp(0)

print ('Training is .....')
print ('iter 			error' )
weights = trainnet(xdata, ydata)

with open('./weights.txt', 'wb') as f:
	pickle.dump([weights[0], weights[1], weights[2], xdata.min(0), xdata.ptp(0)], f)
