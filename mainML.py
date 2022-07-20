# @Time : 2022-07-20 10:44
# @Author : Phalange
# @File : mainML.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba_fast as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
import re

def RemoveWords(line):
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:logistic',
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    #'nthread': 4,
}
plst = list(params.items())


df = pd.read_csv("data/train.csv",sep='\t')
print(df.info())
print(df.describe())
#删除除字母,数字，汉字以外的所有符号
df['clean_review'] = df['text'].apply(RemoveWords)


stopwords = []
with open("./data/stopwords.txt", 'r', encoding='utf-8') as f:
    text = f.readlines()
for each in text:
    stopwords.append(each.strip())

#分词，并过滤停用词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
count = CountVectorizer()
tfidf = TfidfTransformer()

X = count.fit_transform(df['cut_review'])
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.30,random_state=124)
X_train = X
y_train = df['label']


# X_train = df['text'][:50000]
# y_train = df['label'][:50000]
# X_test = df['text'][50000:]
# y_test = df['label'][50000:]

# dtrain = xgb.DMatrix(X_train, y_train) # 生成数据集格式
# num_rounds = 500
# print("training.......")
# model = xgb.train(plst, dtrain, num_rounds) # xgboost模型训练
# # 对测试集进行预测
# print("testing....")
# dtest = xgb.DMatrix(X_test)
# y_pred = model.predict(dtest)


param_grid = {
    'booster': 'gbtree',
    'objective': 'reg:logistic',
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,



}

clf = XGBClassifier(
    silent=0,
    learning_rate=0.01,
    min_child_weight=1,
    max_depth=6,
    gamma=0,
    subsample=1,
    max_delta_step=0,
    colsample_bytree=1,
    reg_lambda=1,
    n_estimators=100,
    seed=100
)
print("Training......")
clf.fit(X_train,y_train)
print("Testing.......")
y_pred = clf.predict(X_test)



# 计算准确率
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))


"""
生成submit.csv
"""
print("生成submit.....")
df2 = pd.read_csv("data/test.csv")
df2['clean_review'] = df2['text'].apply(RemoveWords)
df2['cut_review'] = df2['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
testX = count.transform(df2['cut_review'])
testX = tfidf.transform(testX)
ytest = clf.predict(testX)

df3 = pd.DataFrame(data=ytest,index=None,columns=['label'])
df3.to_csv("sample_submit.csv",index=False,encoding="utf-8-sig")
