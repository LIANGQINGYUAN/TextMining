# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:50:04 2018

@author: liang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_excel('分词结果.xlsx')

X, y = df.iloc[:, 1:], df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#随机选择30%作为测试集，剩余作为训练集


#对训练集进行合并，进行TF-IDF
df_train=pd.DataFrame()
df_train['types']=y_train
df_train['content']=X_train

#对测试集进行合并
df_test=pd.DataFrame()
df_test['types']=y_test
df_test['content']=X_test


#训练集类别分开
catagory01=df_train.loc[df_train['types']==1]['content'].tolist()
catagory02=df_train.loc[df_train['types']==2]['content'].tolist()
catagory03=df_train.loc[df_train['types']==3]['content'].tolist()
catagory04=df_train.loc[df_train['types']==4]['content'].tolist()
catagory05=df_train.loc[df_train['types']==5]['content'].tolist()
catagory06=df_train.loc[df_train['types']==6]['content'].tolist()

corpus=[]
corpus.append(catagory01)
corpus.append(catagory02)
corpus.append(catagory03)
corpus.append(catagory04)
corpus.append(catagory05)
corpus.append(catagory06)

#将测试数据加入已分类的行后
for i in df_test['content']:
    corpus.append(i)

#转化成【‘ ’，‘ ’，‘ ’】
list_str_texts=[]
for i in range(len(corpus)):
    s=str(corpus[i])
    list_str_texts.append(s)
    
#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()   #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer() #该类会统计每个词语的tf-idf权值  

tfidf=transformer.fit_transform(vectorizer.fit_transform(list_str_texts))


#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  

#将每类文档的词频写入文件
for index in range(0, 6):
    with open("tfidf_%d" % index, "w") as f:
    
        for i in np.argsort(-tfidf.toarray()[index]):
            if tfidf.toarray()[index][i] > 0:
                f.write("%f %s\n" % (tfidf.toarray()[index][i], word[i]))
        f.close()


#关键词提取
def feature_extraction():
    d = {}
    for index in range(0, 6):
        with open("tfidf_%d" % index, "r") as f:
            lines = f.readlines()
            for line in lines:
                word = line.split(' ')[1][:-1]
                tfidf = line.split(' ')[0]
                if word in d:
                    d[word] = np.append(d[word], tfidf)
                else:
                    d[word] = np.array(tfidf)
            f.close();
    with open("features.txt", "w") as f:        
        for word in d:
            if d[word].size >= 2 :
                index = np.argsort(d[word])
                if float(d[word][index[d[word].size-0-1]]) - float(d[word][index[d[word].size-1-1]]) > 0.00005:
                    f.write("%s %s\n" % (word, d[word]))
        f.close()
    return d
        
d=feature_extraction() 

with open("features.txt", "r") as f:
    lines = f.readlines()
    print('最终的特征数量：%d',len(lines))
    f.close()


#训练
from sklearn.svm import SVC

#根据提取的关键词删除掉无用的关键词
model = SVC()
y=np.array(range(1,7))
model.fit(tfidf[:6], y)#前6行进行训练
predicted=model.predict(tfidf[6:])#6行以后的进行预测
print('测试准确率：%s'%model.score(tfidf[6:],df_test['types']))#测试准确率

'''
svm优化-抽取特征
'''
df_tfidf=pd.DataFrame(weight)
df_tfidf.cloumns=word
df_tfidf.to_excel('df_tfidf_2.xlsx')
df_word=pd.DataFrame(word)
df_word.to_excel('df_word_2.xlsx')
df_test.to_excel('df_test_2.xlsx')