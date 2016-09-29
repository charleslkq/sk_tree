#coding=utf-8
'''
Created on 2016年9月01日

@author: charles.lee
'''

from sklearn.feature_selection import RFE
from sklearn import decomposition
import os,sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import   train_test_split  #k折交叉模块  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve   
from sklearn.metrics import  classification_report  
from os.path import join
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
def load_data():
	dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
	data = np.loadtxt(join(dirname,'noble_pay_lost1.txt'))
	arrtarget=data[:,-1]
#	sc_data=data[:,2:7]	
#	sc_data=np.array(zip(data[:,0],data[:,2:7]))
	sc_data=np.c_[data[:,0],data[:,2:7]]
	enc = OneHotEncoder(categorical_features=np.array([0]),n_values=[8]) #等级one hot code
	enc.fit(sc_data) 
	train_feature = enc.transform(sc_data).toarray()  	
	return train_feature,arrtarget

  

if __name__=='__main__':
	a,b=load_data()
	x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.2) 
	
#利用信息熵作为划分标准，对决策树进行训练 
	clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=7)   #entropy gini
	print clf  
	clf.fit(x_train,y_train)  
#把决策树写入文件  
	if os.path.isdir(u'D:\\sklearn测试库'):  
		pass  
	else:  
		os.makedirs(u'D:\\sklearn测试库')  
	with open(u'D:\\sklearn测试库\\决策结果.txt','w') as  f:  
		f=tree.export_graphviz(clf,out_file=f)  
	print(u'打印出特征')  
#系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''    
	expected = y_test
	predicted = clf.predict(x_test)
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))

	
	#logisticrgression
	lx_train,lx_test,ly_train,ly_test=train_test_split(a,b,test_size=0.2) 
	#数据标准化正则化
#	min_max_scaler = preprocessing.MinMaxScaler()
#	X_train_minmax = min_max_scaler.fit_transform(lx_train)
#	X_test_minmax = min_max_scaler.transform(lx_test)

	scaler = preprocessing.StandardScaler()
	X_train_scaler = scaler.fit_transform(lx_train)
	X_test_scaler = scaler.transform(lx_test)
	
#	normalizer=preprocessing.Normalizer()
#	nx_train = normalizer.fit_transform(X_train_scaler) 
#	nx_test=normalizer.transform(X_test_scaler)   


	
#特征选择
	model = ExtraTreesClassifier()
	model.fit(X_train_scaler,ly_train)
	print('feature_importances:')
	print(model.feature_importances_)
	
	
	print 'feature normalizer LogisticRegression:'
	ng_model = LogisticRegression(penalty='l1',C=10,tol=0.001,max_iter=100)
	ng_model.fit(X_train_scaler, ly_train)
	nexpected = ly_test
	npredicted = ng_model.predict(X_test_scaler)
	print(metrics.classification_report(nexpected, npredicted))
	print(metrics.confusion_matrix(nexpected, npredicted))

	print(u'参数优化：')  
	tuned_parameters ={'penalty': ['l1','l2'], 'tol': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}
	clf =GridSearchCV(LogisticRegression(), tuned_parameters)
	clf.fit(X_train_scaler, ly_train)
	print(clf.best_estimator_)	
#pca降维
#	x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.2) 
	print 'PCA LogisticRegression:'
	pca = decomposition.PCA(n_components=2)
	pca.fit(X_train_scaler)
	tr_feature=pca.transform(X_train_scaler)
	te_feature=pca.transform(X_test_scaler)
#	px_train,px_test,py_train,py_test=train_test_split(a_pca,b,test_size=0.2) 
	lg_model = LogisticRegression(penalty='l1',C=1000)
	lg_model.fit(tr_feature, ly_train)
	pexpected = ly_test
	ppredicted = lg_model.predict(te_feature)
	print(metrics.classification_report(pexpected, ppredicted))
	print(metrics.confusion_matrix(pexpected, ppredicted))	
	#score_func=metrics.f1_score 可选择指标参数，默认自带的准确率算法。
	scores = cross_validation.cross_val_score(lg_model, te_feature, y_test, cv=5)
	print scores
'''
#交叉认证分
#scores = cross_validation.cross_val_score(clf, raw data, raw target, cv=5, score_func=None)
clf是不同的分类器，可以是任何的分类器。比如支持向量机分类器。clf = svm.SVC(kernel='linear', C=1)
cv参数就是代表不同的cross validation的方法了。如果cv是一个int数字的话，并且如果提供了raw target参数，那么就代表使用StratifiedKFold分类方式，如果没有提供raw target参数，那么就代表使用KFold分类方式。
cross_val_score函数的返回值就是对于每次不同的的划分raw data时，在test data上得到的分类的准确率。至于准确率的算法可以通过score_func参数指定，如果不指定的话，是用clf默认自带的准确率算法。

from sklearn import cross_validation
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)#5-fold cv
#change metrics
from sklearn import metrics
cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, score_func=metrics.f1_score)

'''

'''	
	model = LogisticRegression()
# create the RFE model and select 3 attributes
	rfe = RFE(model, 3)            #暴力破解呀，选出所有大小为3的子集算出误差最小的
	rfe = rfe.fit(a_pca, b)
# summarize the selection of the attributes
	print(rfe.support_)
	print(rfe.ranking_)
'''


'''
from sklearn import decomposition
pca = decomposition.PCA()
PCA(copy=True, n_components=None, whiten=False)
iris_pca = pca.fit_transform(iris_X)
iris_pca[:5]
pca.explained_variance_ratio_

pca = decomposition.PCA(n_components=2)
iris_X_prime = pca.fit_transform(iris_X)
iris_X_prime.shape
#data = np.loadtxt('D:\\YY&work\\python\\scikit-learn\\tree\\noble_pay_lost1.txt')
'''












