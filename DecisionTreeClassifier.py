#encoding:utf-8  
''''' 
Created on 2015年10月14日 
 
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
from sklearn.cross_validation import   train_test_split  #k折交叉模块  
data=[]  
label=[]  
with open(u'D:\\sklearn测试库\\决策树.txt') as f:  
    for i in f:    
        tokens = i.strip().split(' ')    
        data.append([float(tk) for tk in tokens[:-1]])    
        label.append(tokens[-1])    
x=np.array(data)  
label=np.array(label)  
y=np.zeros(label.shape)  
#标签转换为0和1的形式  
y[label=='fat']=1  
#拆分数据  
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
'''''系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''    
print(clf.feature_importances_)  
answer=clf.predict(x_train)  
print(u'系统进行测试')  
print(x_train)  
print(answer)  
print(y_train)  
print(np.mean(answer==y_train))  
  
''''''准确率和召回率'''  
precision,recall,thresholds=precision_recall_curve(y_train, clf.predict(x_train))  
print(u'准确情况')  
print(precision)  
print(recall)  
print(thresholds)  
answer=clf.predict_proba(x)[:,1]  
print(u'预测值')  
print(clf.predict_proba(x))  
print(classification_report(y, answer, target_names = ['thin', 'fat']))    
  