# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:24:30 2018

@author: liang
"""
# -*- coding: utf-8 -*-

import jieba
import pandas as pd
import re

'''
读取数据
'''
df = pd.read_excel('原始整理数据（改）.xlsx',encoding = "gb18030")

'''
分词
'''
mycut=lambda s:' '.join(jieba.cut(s))
documents=df["自述"].apply(mycut)
#去停用词
import codecs
with codecs.open("stopwords.txt", "r", encoding="utf-8") as f:
    text = f.read()
stoplists=text.splitlines()
texts = [[word for word in document.split()if word not in stoplists] for document in documents]

list_str_texts=[]

for i in range(len(texts)):
    s=str(texts[i])
    s=re.sub("[',]",'',s)
    list_str_texts.append(s)

str_texts=pd.DataFrame(list_str_texts)
str_texts['types']=df['类别']
str_texts.to_excel('分词结果.xlsx')

'''
计算词频 TF-IDF
'''
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()   #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer() #该类会统计每个词语的tf-idf权值  

tfidf=transformer.fit_transform(vectorizer.fit_transform(list_str_texts))


#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  

df_tfidf=pd.DataFrame(weight)
df_tfidf.cloumns=word
df_tfidf.to_excel('df_tfidf.xlsx')
df_word=pd.DataFrame(word)
df_word.to_excel('df_word.xlsx')

df_data = pd.read_excel('df_tfidf.xlsx')
df_data['types']=df['类别']
x=df_data.iloc[:,:-1]
y=df_data['types']

'''
抽取重要特征
'''
from sklearn.ensemble import ExtraTreesClassifier 
model=ExtraTreesClassifier(n_estimators=57,
                                criterion='entropy',
                                min_impurity_decrease=93*0.00001,
                                max_depth=33,
                                min_samples_split=27,
                                min_samples_leaf=3,max_leaf_nodes=47,
                                class_weight='balanced') 
model.fit(x,y)
importances=model.feature_importances_

data_extracted=pd.DataFrame()
for i in range(len(importances)):
    if importances[i]!=0:
        data_extracted[i]=df_data[i]

data_extracted['types']=df_data['types']


'''
训练
'''
#分训练集和测试集
from sklearn.model_selection import train_test_split

X, y = data_extracted.iloc[:, :-1], data_extracted['types']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#随机选择30%作为测试集，剩余作为训练集

#SVM训练
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print('svc训练准确率：%s'%svc.score(X_train,y_train))#测试准确率

#决策树训练
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

#贝叶斯训练
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高  
clf = MultinomialNB(alpha=0.01).fit(X_train, y_train)  

#预测分类结果  
predicted = clf.predict(X_test)  
# 计算分类精度：  
from sklearn import metrics  
def metrics_result(actual, predict):  
    print('贝叶斯结果：')
    print ('精度:{0:.3f}'.format(metrics.precision_score(actual, predict,average='weighted')) ) 
    print ('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict,average='weighted'))   )
    print ('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict,average='weighted'))   )

metrics_result(y_test,predicted)