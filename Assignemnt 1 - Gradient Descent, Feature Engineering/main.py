import numpy as np
from copy import deepcopy

iterations = 3000000
numiter = 2000 # for calculating maxlearnrate
#learning_rate = 0.000003206
epsilon = 20
XMTotal = np.loadtxt(open("data/train.csv", "rb"), delimiter=",", skiprows=1,usecols=range(1,15))
XM = XMTotal[ : , 0:13]
X2M = np.loadtxt(open("data/test.csv", "rb"), delimiter=",", skiprows=1,usecols=range(1,14))
X2T = np.append(X2M, np.ones( (105,1) ),axis = 1 )
#XM[:,9] = XM[:,9]**2
numrows , numattr = np.shape(XM)
#lamda = 0 # got from k fold for cross validation for seed 6181 and cost 22.73
X1mm = np.append(XM, np.ones( (numrows,1) ),axis = 1 )# added 1 vector for b value
numattr += 1
X=X1mm[0:400,:]						#**


X2=X1mm[360:400,:]

Y1=XMTotal[ :, 13]

Y=Y1[0:400]
Y2=Y1[360:400]

#########-###-------------##############----------------################-------------###############
lamda = 0.001

#  ------	1
X = np.c_[ X,  np.log( X[:,0]) ]
X = np.c_[ X,  np.power( X[:,0],0.36) ]
X2T = np.c_[ X2T,  np.log( X2T[:,0]) ]
X2T = np.c_[ X2T,  np.power( X2T[:,0],0.36) ]
#			2
X = np.c_[ X,  np.power( X[:,1],1.5) ]
X = np.c_[ X,  np.power( X[:,1],1.4) ]
X = np.c_[ X,  np.power( X[:,1],1e-4) ]
X2T = np.c_[ X2T,  np.power( X2T[:,1],1.5) ]
X2T = np.c_[ X2T,  np.power( X2T[:,1],1.4) ]
X2T = np.c_[ X2T,  np.power( X2T[:,1],1e-4) ]
##			3
X = np.c_[ X,  np.power( X[:,2],1e-5) ]
X2T = np.c_[ X2T,  np.power( X2T[:,2],1e-5) ]
##			5
X = np.c_[ X,  np.log( X[:,4]) ]
X2T = np.c_[ X2T,  np.log( X2T[:,4]) ]
##			6
X = np.c_[ X,  np.power( X[:,5],0.5) ]
X2T = np.c_[ X2T,  np.power( X2T[:,5],0.5) ]
##			7
X = np.c_[ X,  np.power( X[:,6],1.1) ]
X2T = np.c_[ X2T,  np.power( X2T[:,6],1.1) ]
##			8
X = np.c_[ X,  np.log( X[:,7]) ]
X2T = np.c_[ X2T,  np.log( X2T[:,7]) ]
X = np.c_[ X,  np.power( X[:,7],0.001) ]
X2T = np.c_[ X2T,  np.power( X2T[:,7],0.001) ]
X = np.c_[ X,  np.power( X[:,7],0.01) ]
X2T = np.c_[ X2T,  np.power( X2T[:,7],0.01) ]
###			9
X = np.c_[ X,  np.power( 1 / X[:,8],12) ]
X = np.c_[ X,  np.log( X[:,8]) ]
X2T = np.c_[ X2T,  np.power( 1 / X2T[:,8],12) ]
X2T = np.c_[ X2T,  np.log( X2T[:,8]) ]
X = np.c_[ X,  np.power((1/ X[:,8]),10) ]
X2T = np.c_[ X2T,  np.power( 1/X2T[:,8],10) ]
X = np.c_[ X,  np.power((1/ X[:,8]),8) ]
X2T = np.c_[ X2T,  np.power( 1/X2T[:,8],8) ]
##			10
X = np.c_[ X,  np.log( X[:,9]) ]
X2T = np.c_[ X2T,  np.log( X2T[:,9]) ]
X = np.c_[ X,  np.power( X[:,9],0.1) ]
X2T = np.c_[ X2T,  np.power( X2T[:,9],0.1) ]
##			11
X = np.c_[ X,  np.power(X[:,10],1.5) ]
X2T = np.c_[ X2T,  np.power(X2T[:,10],1.5) ]
##			12
X = np.c_[ X,  np.exp( 2/X[:,11]) ]
X2T = np.c_[ X2T,  np.exp( 2/X2T[:,11]) ]

#X = np.delete(X,[7], 1)
#X2T = np.delete(X2T,[7], 1)


numattr = np.shape(X)[1]
#print(X2)
#print(Y2)


Xtrans = X.transpose()				#**
XtransX = np.dot(Xtrans,X)			#**
XtransY = np.dot(Xtrans,Y)			#**
#W = np.zeros(numattr)
		
def errorw(W,p,lamda):
	return np.sum((np.dot(X,W)-Y)**2) + lamda*np.sum(np.power(np.absolute(W),p))

def grad_of_errorw(W,p,lamda):
	#print (p)
	return ((2/numrows)*(np.dot(XtransX,W)- XtransY) + p*lamda*np.sign(W)*np.power(np.absolute(W),(p-1)) )

def KfoldErrow(Xk1,Yk1 ,W,p,lamda):
	return np.sum((np.dot(Xk1,W)-Yk1)**2) + lamda*np.sum(np.power(np.absolute(W),p ) )

def KfoldGrad_Error_w(Xk1,Yk1 ,W,p,lamda):
	return ((2/numrows)*(np.dot(np.dot(Xk1.transpose(),Xk1) ,W)- np.dot(Xk1.transpose(),Yk1)) + p*lamda*np.sign(W)*np.power(np.absolute(W),(p-1)) )

def maxratecal(W,p,lamda):
	#Xtrans = X.transpose()	
	epreached = False
	i = 0
	cost = 10
	precost = 0
	maxlearnrate = -1.0
	minlearnrate = 10
	learning_rate= 0
	precosti =600000
	while((not epreached) and (i < numiter)) :
		learning_rate =1000
		while(errorw(W - (learning_rate / 2)*grad_of_errorw(W, p,lamda), p,lamda) < errorw(W - learning_rate*grad_of_errorw(W, p,lamda), p,lamda) ):
			learning_rate = learning_rate /2
		if(maxlearnrate < learning_rate):
			maxlearnrate = learning_rate
		if(minlearnrate > learning_rate):
			minlearnrate = learning_rate
		cost = np.sum((np.dot(X, W) - Y)**2) / (1* numrows)
		#print (i,'	,	',cost,'	,	',(precost -cost),'	max , min->  ',maxlearnrate , '  ,  ', minlearnrate )
		if(cost == precost):
			break
		precost = cost
		W = W - (learning_rate)*grad_of_errorw(W,p,lamda)
		if cost < epsilon*1.2:
			epreached = True
			print('epsilon reached ')
			print ('numsteps ', i)
			break
		i +=1

	return maxlearnrate , minlearnrate

def Kfoldfitting(Xk ,Yk, W,learning_rate_intial,p,lamda):
	epreached = False
	i = 0
	cost = 10
	precost = 0
	maxlearnrate = -1.0
	learning_rate= 0
	minlearnrate = 10
	kfolditer = 10000
	#poin1i = 0
	precosti =1000000
	while((not epreached) and (i < kfolditer)) :
		learning_rate = learning_rate_intial
		while(KfoldErrow(Xk,Yk ,W - (learning_rate / 2)*KfoldGrad_Error_w(Xk,Yk ,W, p,lamda), p,lamda) < KfoldErrow(Xk,Yk ,W - learning_rate*KfoldGrad_Error_w(Xk,Yk ,W, p,lamda), p,lamda) ):
			learning_rate = learning_rate /2
		cost = np.sum((np.dot(X, W) - Y)**2) / (1* numrows)
		#print (i,'	,	',cost,'	,	',(precost -cost),'	max , min->  ',maxlearnrate , '  ,  ', minlearnrate )
		if(cost == precost):
			break
		precost = cost
		W = W - (learning_rate)*KfoldGrad_Error_w(Xk,Yk,W,p,lamda)
		if cost < epsilon:
			epreached = True
			#print('epsilon reached ')
			#print ('numsteps ', i)
			break
		i +=1
	
	return W

# 10 fold cross validation
def KfoldCost(X1,lamda,numattr,numrows,lrate,p):
	Y_Test = np.ones((100 ,1))
	X_Test =np.ones((100  ,numattr))
	Y_Train= np.ones((300 ,1))
	X_Train =np.ones((300 ,numattr))
	#print ('x1 ', X1)
	avgcost = 0
	for i in range(1,5):
		#print ('i' , i)
		Y_Test= X1[( (i-1)*100) : (i*100) , numattr]
		X_Test = X1[( (i-1)*100) : (i*100) ,0: numattr]
		if (i != 1) | (i != 4) :
			X_Train =np.append(X1[ : ((i-1)*100) ,0:numattr], X1[ (i*100) : ,0:numattr ],axis = 0 )
			Y_Train = np.append(X1[ : ((i-1)*100) , numattr], X1[ (i*100) : , numattr],axis = 0 )
		else:
			if(i == 1):
				X_Train=X1[(i*100) : , 0:numattr]
				Y_Train=X1[(i*100) : , numattr]
			else:
				X_Train =X1[ : ((i-1)*100) ,0:numattr] 
				Y_Train =X1[ : ((i-1)*100) ,numattr]
		#
		W1 = deepcopy(np.zeros(numattr))
		#w = (np.linalg.inv( (X_Train.transpose).dot(X_Train) +  np.ones((numattr,numattr) ) ) ).dot((X_Train.transpose).dot(Y))
		# X
		#w1 = np.dot((X_Train.transpose()),X_Train)
		#w2 = np.linalg.inv( w1 + (lamda * np.ones((numattr,numattr))))
		#w = np.dot(w2, np.dot(X_Train.transpose(),Y_Train))
		w = Kfoldfitting(X_Train,Y_Train, W1,lrate,p,lamda)
		avgcost += (np.sum((np.dot(X_Test, w) - Y_Test)**2) / (numrows))


	return avgcost

def kfoldMatrix(Xinput,numattr,numrows,seedval,lrate,p):
	X1 = np.array(Xinput)
	np.random.seed(seedval)
	np.random.shuffle(X1)
	lamdarray = [10,0.001,0.005,0.01,0.1,1,2]
	minavg = 100000000
	avg = 0
	minlamda = -1
	#print ('lambda		,		avgcost')
	for lamda in lamdarray:
		avg = KfoldCost(X1,lamda,numattr,numrows,lrate,p)
		print (lamda,'			',avg)
		if(avg < minavg):
			minavg = avg
			minlamda = lamda
			print (minlamda)
	return minlamda , minavg


def fitting(W,learning_rate_intial,p,lamda):
	#Xtrans = X.transpose()	
	epreached = False
	i = 0
	cost = 10
	precost = 0
	maxlearnrate = -1.0
	learning_rate= 0
	minlearnrate = 10

	#poin1i = 0
	precosti =6000000
	while((not epreached) and (i < iterations)) :
		learning_rate = learning_rate_intial
		while(errorw(W - (learning_rate / 2)*grad_of_errorw(W, p,lamda), p,lamda) < errorw(W - learning_rate*grad_of_errorw(W, p,lamda), p,lamda) ):
			learning_rate = learning_rate /2
		if(maxlearnrate < learning_rate):
			maxlearnrate = learning_rate
		if(minlearnrate > learning_rate):
			minlearnrate = learning_rate
		cost = np.sum((np.dot(X, W) - Y)**2) / (1* numrows)
		#print (i,'	,	',cost,'	,	',(precost -cost),'	max , min->  ',maxlearnrate , '  ,  ', minlearnrate )
		if(cost == precost):
			break
		precost = cost
		W = W - (learning_rate)*grad_of_errorw(W,p,lamda)
		if cost < epsilon:
			epreached = True
			print('epsilon reached ')
			print ('numsteps ', i)
			break
		i +=1
	
	return W , cost # temporaray

def main():

	W = np.zeros(numattr)
	lamda = 1
	maxlrate , minlrate = maxratecal(W,2,lamda)
	print( 'lrate is  ',maxlrate)
	W = deepcopy(np.zeros(numattr))
	avcost = 0
	#for p in p_array:
	lamda1 , avcost = kfoldMatrix(np.c_[X,Y],numattr,numrows ,0 ,2* minlrate, 2)
	Wstar ,costto  =  fitting(W,2*minlrate,2,lamda1)
	#print ('wtar is ' , Wstar)
	print ('for p=2 , min lamda =',lamda1, ' avg cost is = ',avcost,'.' ,'cost is ' ,(np.sum((np.dot(X, Wstar) - Y)**2) / ( numrows)) )

	# predict for tests
	
	Y2TM = np.ones((105,2))
	Y2T = X2T.dot(Wstar)
	Y2TM[:,0] = np.arange(0,105).astype(int)
	Y2TM[:,1] = Y2T
	np.savetxt("output.csv", Y2TM, delimiter=",",fmt="%i , %1.9f", header="ID,MEDV", comments="")

	W = deepcopy(np.zeros(numattr))

	Wp1 = deepcopy(np.zeros(numattr))
	#maxlrate , minlrate = maxratecal(W,1.2,lamda)
	print( 'lrate is  ',maxlrate)
	Wp1 = deepcopy(np.zeros(numattr))
	lamda1 , avcost = kfoldMatrix(np.c_[X,Y],numattr,numrows ,0,maxlrate,1.2)
	#print('for p=' ,1.2 ,' min lamda =',lamda, ' avg cost is = ',avcost,'.'  )
	Wstar ,costto  =  fitting(Wp1,2*minlrate,1.2,lamda1)
	#print ('wtar is ' , Wstar)
	print ('for p= 1.2 , min lamda =',lamda1, ' avg cost is = ',avcost,'.' ,'cost is ' ,(np.sum((np.dot(X, Wstar) - Y)**2) / ( numrows)) )

	Y2TM = np.ones((105,2))
	Y2T = X2T.dot(Wstar)
	Y2TM[:,0] = np.arange(0,105).astype(int)
	Y2TM[:,1] = Y2T
	np.savetxt("output_p1.csv", Y2TM, delimiter=",",fmt="%i , %1.9f" ,header="ID,MEDV", comments="")
	
	Wp2 = deepcopy(np.zeros(numattr))
	W = deepcopy(np.zeros(numattr))

	#maxlrate , minlrate = maxratecal(W,1.2,lamda)
	print( 'lrate is  ',maxlrate)
	Wp2 = deepcopy(np.zeros(numattr))
	lamda1 , avcost = kfoldMatrix(np.c_[X,Y],numattr,numrows ,0,maxlrate,1.5)
	Wstar ,costto  =  fitting(Wp2,2*minlrate,1.5,lamda1)
	#print ('wtar is ' , Wstar)
	print ('for p= 1.5 , min lamda =',lamda1, ' avg cost is = ',avcost,'.' ,'cost is ' ,(np.sum((np.dot(X, Wstar) - Y)**2) / ( numrows)) )

	Y2TM = np.ones((105,2))
	Y2T = X2T.dot(Wstar)
	Y2TM[:,0] = np.arange(0,105).astype(int)
	Y2TM[:,1] = Y2T
	np.savetxt("output_p2.csv", Y2TM, delimiter=",", fmt="%i , %1.9f",header="ID,MEDV", comments="")

	Wp3 = deepcopy(np.zeros(numattr))
	W = deepcopy(np.zeros(numattr))
	#maxlrate , minlrate = maxratecal(W,1.2,lamda)
	print( 'lrate is  ',maxlrate)
	Wp3 = deepcopy(np.zeros(numattr))
	lamda1 , avcost = kfoldMatrix(np.c_[X,Y],numattr,numrows ,0,maxlrate,1.8)
	Wstar ,costto  =  fitting(Wp3,2*minlrate,1.8,lamda1)
	print ('for p= 1.8 , min lamda =',lamda1, ' avg cost is = ',avcost,'.' ,'cost is ' ,(np.sum((np.dot(X, Wstar) - Y)**2) / ( numrows)) )

	Y2TM = np.ones((105,2))
	Y2T = X2T.dot(Wstar)
	Y2TM[:,0] = np.arange(0,105).astype(int)
	Y2TM[:,1] = Y2T
	np.savetxt("output_p3.csv", Y2TM, delimiter=",", fmt="%i , %1.9f",header="ID,MEDV", comments="")
		
main()