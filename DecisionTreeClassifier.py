#encoding:utf-8  
''''' 
Created on 2015��10��14�� 
 
@author: ZHOUMEIXU204 
'''  
# data:  
# 1.5 50 thin  
# 1.5 60 fat  
# 1.6 40 thin  
# 1.6 60 fat  
# 1.7 60 thin  
# 1.7 80 fat  
# 1.8 60 thin  
# 1.8 90 fat  
# 1.9 70 thin  
# 1.9 80 fat  
import os  
import numpy as  np  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve   
from sklearn.metrics import  classification_report  
from sklearn.cross_validation import   train_test_split  #k�۽���ģ��  
data=[]  
label=[]  
with open(u'D:\\sklearn���Կ�\\������.txt') as f:  
    for i in f:    
        tokens = i.strip().split(' ')    
        data.append([float(tk) for tk in tokens[:-1]])    
        label.append(tokens[-1])    
x=np.array(data)  
label=np.array(label)  
y=np.zeros(label.shape)  
#��ǩת��Ϊ0��1����ʽ  
y[label=='fat']=1  
#�������  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)  
#������Ϣ����Ϊ���ֱ�׼���Ծ���������ѵ��  
clf=tree.DecisionTreeClassifier(criterion='entropy')  
print clf  
clf.fit(x_train,y_train)  
#�Ѿ�����д���ļ�  
if os.path.isdir(u'D:\\sklearn���Կ�'):  
    pass  
else:  
    os.makedirs(u'D:\\sklearn���Կ�')  
with open(u'D:\\sklearn���Կ�\\���߽��.txt','w') as  f:  
    f=tree.export_graphviz(clf,out_file=f)  
print(u'��ӡ������')  
'''''ϵ����ӳÿ��������Ӱ������Խ���ʾ�������ڷ������𵽵�����Խ�� '''    
print(clf.feature_importances_)  
answer=clf.predict(x_train)  
print(u'ϵͳ���в���')  
print(x_train)  
print(answer)  
print(y_train)  
print(np.mean(answer==y_train))  
  
''''''׼ȷ�ʺ��ٻ���'''  
precision,recall,thresholds=precision_recall_curve(y_train, clf.predict(x_train))  
print(u'׼ȷ���')  
print(precision)  
print(recall)  
print(thresholds)  
answer=clf.predict_proba(x)[:,1]  
print(u'Ԥ��ֵ')  
print(clf.predict_proba(x))  
print(classification_report(y, answer, target_names = ['thin', 'fat']))    
  