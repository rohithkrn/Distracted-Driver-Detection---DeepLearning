import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#mymethod=" "
def my_cross_val(method,X,y,k):
    if(method=="LinearSVC"):
        mymethod=LinearSVC()
    if(method=="SVC"):
        mymethod=SVC()
    if (method=="LogisticRegression"):
        mymethod=LogisticRegression(penalty='l2')
    n_samples,n_features=X.shape
    
    error_rates=np.zeros(k)
    Xsplit=np.array_split(X, k)
    ysplit=np.array_split(y,k)
    for i in range(k):
        Xtest=Xsplit[i]
        ytest=ysplit[i]
        Xtrain=np.zeros((1,n_features))
        ytrain=np.zeros((1,1))
        for j in range(k):
            if j!=i:
#                
                Xtrain=np.vstack((Xtrain,Xsplit[j]))
                ytrain=np.vstack((ytrain,(ysplit[j].reshape((-1,1)))))
              
        Xtrain=np.delete(Xtrain,0,0) 
        ytrain=np.delete(ytrain,0,0) 
        Xtrain1=Xtrain
        ytrain1=ytrain
        mymethod.fit(Xtrain1,ytrain1)
        result= mymethod.predict(Xtest)

        count=0
        for m in range(len(result)):
            if (ytest[m]!=result[m]):
                count=count+1
        error_rates[i]=float(count)/(len(result))
        
               
    mean_error_rate=np.mean(error_rates) 
    std_deviation_error_rate=np.std(error_rates) 
        
    if(method=="LinearSVC"):
            print "Error rates for LinearSVC=",error_rates
            print "Mean of Error rates for LinearSVC=",mean_error_rate
            print "Std deviation of Error rates for LinearSVC=",std_deviation_error_rate
            return error_rates,mean_error_rate,std_deviation_error_rate
         
    if(method=="SVC"):
            print "Error rates for SVC=",error_rates
            print "Mean of Error rates for SVC=",mean_error_rate
            print "Std deviation of Error rates for SVC=",std_deviation_error_rate
            return error_rates,mean_error_rate,std_deviation_error_rate
        
    if(method=="LogisticRegression"):
            print "Error rates for LogisticRegression=",error_rates
            print "Mean of Error rates for LogisticRegression=",mean_error_rate
            print "Std deviation of Error rates for LogisticRegression=",std_deviation_error_rate
            return error_rates,mean_error_rate,std_deviation_error_rate 
              
              