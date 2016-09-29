#coding=utf-8
from os.path import dirname
from sklearn.preprocessing import scale
import numpy as np
import csv
from sklearn.cross_validation import   train_test_split  #k折交叉模块  
from sklearn.cross_validation import   train_test_split  #k折交叉模块  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve   
from sklearn.metrics import  classification_report  
import os
def load_data():
	with open('D:\\YY&work\\python\\scikit-learn\\tree\\noble_pay_lost.csv')as csv_file:
		data_file = csv.reader(csv_file)
#		temp = next(data_file)
		data=[]
		target=[]
		for  ir in data_file:
			data.append([ir[0],ir[2],ir[3],ir[4],ir[5],ir[6]])
			target.append(ir[-1])
	arrdata=np.array(data)
	arrtarget=np.array(target)
	rmf=scale(arrdata[:,1:4])
	rmf_score=np.sum(np.array(zip(rmf[:,0]*3,rmf[:,1]*2,rmf[:,2]),dtype=np.float),axis=1)
	f_data=np.array(zip(arrdata[:,0],rmf_score,arrdata[:,4],arrdata[:,5]))
	return f_data,arrtarget
	
if __name__=='__main__':
	x,y=load_data()
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) 
#利用信息熵作为划分标准，对决策树进行训练 
	clf=tree.DecisionTreeClassifier(criterion='entropy')  
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
	print(clf.feature_importances_)  
	answer=clf.predict(x_train)  
	print(u'系统进行测试')  
	print(x_train)  
	print(answer)  
	print(y_train)  
	print(np.mean(answer==y_train)) 
	r = clf.score(x_test , y_test)
	print r
'''	
	print 'precision,recall'  
#准确率和召回率
	precision,recall,thresholds=precision_recall_curve(y_train, clf.predict(x_train))  
	print(u'准确情况')  
	print(precision)  
	print(recall)  
	print(thresholds)  
	answer=clf.predict_proba(x)[:,1]  
	print(u'预测值')  
	print(clf.predict_proba(x))  
	print(classification_report(y, answer, target_names = ['thin', 'fat']))    
'''


