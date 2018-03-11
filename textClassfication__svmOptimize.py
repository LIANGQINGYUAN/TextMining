# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 08:52:18 2018

@author: liang
"""
#训练
from sklearn.svm import SVC
import numpy as np
import pandas as pd

model = SVC(C=2, kernel='rbf', decision_function_shape='ovr')
y=np.array(range(1,7))

#读入数据
df_tfidf=pd.read_excel('df_tfidf_2.xlsx')
df_test=pd.read_excel('df_test_2.xlsx')
df_tfidf_extracted=pd.DataFrame()

#选择提取出来的特征
with open("features.txt", "r") as f:
    lines = f.readlines()
    print('最终的特征数量：%d',len(lines))
    for line in lines:
        word_extracted = line.split(' ')[0]
        #删除不在抽取出来的特征里面的无关特征
        if word_extracted  in df_tfidf.columns.values:
            df_tfidf_extracted[word_extracted]=df_tfidf[word_extracted]
    f.close()

df_tfidf_extracted.to_excel('df_tfidf_extracted_2.xlsx')

#训练
model.fit(df_tfidf_extracted[:6], y)#前6行进行训练
predicted=model.predict(df_tfidf_extracted[6:])#6行以后的进行预测
print('训练准确率-优化后：%s'%model.score(df_tfidf_extracted[:6],y))#测试准确率
print('测试准确率-优化后：%s'%model.score(df_tfidf_extracted[6:],df_test['types']))#测试准确率


X_train=df_tfidf_extracted[:6]
y_train=y
X_test=df_tfidf_extracted[6:]
y_test=df_test['types']
#贝叶斯训练
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高  
clf = MultinomialNB(alpha=0.01).fit(X_train, y_train)  

#预测分类结果  
predicted = clf.predict(X_test)  
# 计算分类精度：  
from sklearn import metrics  
def metrics_result(actual, predict):  
    print ('精度:{0:.3f}'.format(metrics.precision_score(actual, predict,average='weighted')) ) 
    print ('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict,average='weighted'))   )
    print ('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict,average='weighted'))   )

metrics_result(y_test,predicted)

#决策树训练
from sklearn.ensemble import ExtraTreesClassifier 
extraTrees=ExtraTreesClassifier(n_estimators=57,
                                criterion='entropy',
                                min_impurity_decrease=93*0.00001,
                                max_depth=33,
                                min_samples_split=27,
                                min_samples_leaf=3,max_leaf_nodes=47,
                                class_weight='balanced') 
extraTrees.fit(X_train, y_train)

print('extraTrees训练准确率：%s'%(extraTrees.score(X_train,y_train)))
print('***********extraTrees测试准确率***********：%s'%(extraTrees.score(X_test,y_test)))