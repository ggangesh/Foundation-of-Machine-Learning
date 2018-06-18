import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import SGDClassifier


#features are mapped with weights
workclassd = {'Private': 3, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':2, 'Local-gov':1, 'State-gov':1, 'Without-pay':0, 'Never-worked':0,'?':0}
educationd = {'Bachelors':8, 'Some-college':7, '11th':5, 'HS-grad':4, 'Prof-school':9, 'Assoc-acdm':10, 'Assoc-voc':9, '9th':3, '7th-8th':2, '12th':6, 'Masters':11, '1st-4th':0, '10th':4, 'Doctorate':12, '5th-6th':1, 'Preschool':0,'?':0}
maritalstatusd = { 'Married-civ-spouse':5, 'Divorced':3, 'Never-married':2, 'Separated':1, 'Widowed':4, 'Married-spouse-absent':0, 'Married-AF-spouse':0,'?':0}
occupationd =  {'Tech-support':1, 'Craft-repair':0, 'Other-service':0, 'Sales':3, 'Exec-managerial':5, 'Prof-specialty':4, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':1,'?':0}
relationshipd = {'Wife':2, 'Own-child':0, 'Husband':3, 'Not-in-family':0, 'Other-relative':0, 'Unmarried':1,'?':0}
raced = {'White':2, 'Asian-Pac-Islander':0, 'Amer-Indian-Eskimo':0,'Other':0, 'Black':1,'?':0}
sexd = {'Female':1, 'Male':2,'?':0}
countryd = {'United-States':5, 'Cambodia':0, 'England':3, 'Puerto-Rico':0, 'Canada':4, 'Germany':4, 'Outlying-US(Guam-USVI-etc)':0, 'India':1, 'Japan':3, 'Greece':2, 'South':0, 'China':2, 'Cuba':0, 'Iran':2, 'Honduras':0, 'Philippines':0, 'Italy':3, 'Poland':3, 'Jamaica':1, 'Vietnam':2, 'Mexico':1, 'Portugal':3, 'Ireland':3, 'France':3, 'Dominican-Republic':1, 'Laos':0, 'Ecuador':1, 'Taiwan':2, 'Haiti':0, 'Columbia':1, 'Hungary':1, 'Guatemala':0, 'Nicaragua':0, 'Scotland':2, 'Thailand':2, 'Yugoslavia':2, 'El-Salvador':1, 'Trinadad&Tobago':3, 'Peru':1, 'Hong':1, 'Holand-Netherlands':4,'?':0}


def generateFeatures(data):
	
#End of function generateFeatures

data = pd.read_csv('./train.csv', delimiter=",")
data2 = pd.read_csv('./kaggle_test_data.csv', delimiter=",")
for i in range(len(data)):
	data[i][2] = workclassd[data[i][2].lstrip()]
	data[i][4] = educationd[data[i][4].lstrip()]
	data[i][6] = maritalstatusd[data[i][6].lstrip()]
	data[i][7] = occupationd[data[i][7].lstrip()]
	data[i][8] = relationshipd[data[i][8].lstrip()]
	data[i][9] = raced[data[i][9].lstrip()]
	data[i][10] = sexd[data[i][10].lstrip()]
	data[i][14] = countryd[data[i][14].lstrip()]

for i in range(len(data2)):
	data2[i][2] = workclassd[data2[i][2].lstrip()]
	data2[i][4] = educationd[data2[i][4].lstrip()]
	data2[i][6] = maritalstatusd[data2[i][6].lstrip()]
	data2[i][7] = occupationd[data2[i][7].lstrip()]
	data2[i][8] = relationshipd[data2[i][8].lstrip()]
	data2[i][9] = raced[data2[i][9].lstrip()]
	data2[i][10] = sexd[data2[i][10].lstrip()]
	data2[i][14] = countryd[data2[i][14].lstrip()]

xdata1 = data.drop(data.columns[-1], axis=1)
ydata1 = [value[0] for value in (data.drop(data.columns[0:-1], axis=1)).values]
xdata1 = xdata1.values
xdata1 = np.array(xdata1, dtype=np.int64)
xdata1 = (xdata1 - xdata1.min(0))/xdata1.ptp(0)
xdata1list = xdata1.tolist()

xdata2 = data2.values
ids = (data2.drop(data2.columns[1:], axis=1)).values
xdata2 = np.array(xdata2, dtype=np.int64)
xdata2 = (xdata2 - xdata2.min(0))/xdata1.ptp(0)
xdata2list = xdata2.tolist()

print 'Used linearSVC ,SGD with hinge loss and l2 penalty , SGD with log loss '
lg = SGDClassifier(loss="log").fit(xdata1list, ydata1)
hinge = SGDClassifier(loss="hinge", penalty="l2").fit(xdata1list, ydata1)
linear = svm.LinearSVC().fit(xdata1list, ydata1)

ydata2log = lg.predict(xdata2list)
ydata2hinge = hinge.predict(xdata2list)
ydata2linear = linear.predict(xdata2list)

outlog = np.array([[x] for x in ydata2log])
outhinge = np.array([[x] for x in ydata2hinge])
outlinear = np.array([[x] for x in ydata2linear])

np.savetxt('predictions_1.csv', np.concatenate((ids, outlinear), axis=1), fmt='%d,%.9f', header="id,salary", comments='')
np.savetxt('predictions_2.csv', np.concatenate((ids, outlog), axis=1), fmt='%d,%.9f', header="id,salary", comments='')
np.savetxt('predictions_3.csv', np.concatenate((ids, outhinge), axis=1), fmt='%d,%.9f', header="id,salary", comments='')
