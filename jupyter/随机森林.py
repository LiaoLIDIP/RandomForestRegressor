#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
# import pydot
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pylab as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"/Users/liaolong/Downloads/simsun/simsun.ttc", size=12)

# import xgboost as xgb
# from xgboost import plot_importance

# https://blog.csdn.net/qq_40229367/article/details/88526749


# # 数据处理

# In[2]:


df_data = pd.read_excel('../data/实验数据.xlsx')
df_data['自变量7'] = 0
k = 0
for (index,i) in zip(df_data.index,df_data['时间顺序']):
    if i == 1:
        k += 1
    df_data['自变量7'][index] = k
    if isinstance(df_data['自变量4'][index],str):
        x4 = df_data['自变量4'][index]
        x4 = re.split("°|ˊ",x4)
        x4[-1] = x4[-1][:-1]
        x4_f = float(x4[0]) + float(x4[1])/60 + float(x4[2])/3600
        df_data['自变量4'][index] = x4_f
    else:
        df_data['自变量4'][index] = float(df_data['自变量4'][index])
df_data.drop(['时间顺序'],axis=1,inplace=True,)
df_data['自变量1'][366] = df_data['自变量1'][367]
df_data.columns = ['道路纵坡','曲线半径','曲线长度','路线转角','出口线形', '初始速度','因变量','道路编号']
df_data


# In[4]:


labels = np.array(df_data['因变量','道路编号'])
features = df_data.drop(['因变量','道路编号'], axis=1)
feature_list = list(features)
features = np.array(features)
# print(labels)
# print(features)


# In[78]:


# 数据切分
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.15,random_state = 42) #测试集比例 15%
 
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[ ]:





# In[79]:


# # 数据可视化
 
# # Set the style
# plt.style.use('fivethirtyeight')
 
# # Set up the plotting layout
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
# fig.autofmt_xdate(rotation = 45)
 
# # Actual max temperature measurement
# ax1.plot(dates, features['自变量1'])
# ax1.set_xlabel(''); ax1.set_ylabel('Variable 1'); ax1.set_title('Max Temp')
 
# # Temperature from 1 day ago
# ax2.plot(dates, features['自变量2'])
# ax2.set_xlabel(''); ax2.set_ylabel('Variable 2'); ax2.set_title('Previous Max Temp')
 
# # Temperature from 2 days ago
# ax3.plot(dates, features['自变量3'])
# ax3.set_xlabel('Date'); ax3.set_ylabel('Variable 3'); ax3.set_title('Two Days Prior Max Temp')
 
# # Friend Estimate
# ax4.plot(dates, features['自变量4'])
# ax4.set_xlabel('Date'); ax4.set_ylabel('Variable 4'); ax4.set_title('Friend Estimate')
 
# plt.tight_layout(pad=2)


# # 随机森林

# In[80]:


# 建模，训练

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
 
# Instantiate model 
# n_estimators 森林中树的树量
# max_depth 树的最大深度
rf = RandomForestRegressor(n_estimators=1000, max_depth = 10, random_state=42)
# rf = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False)
 
# Train the model on training data
rf.fit(train_features, train_labels)

# plot_importance(rf)


# In[81]:


# 模型测试

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
 
# Calculate the absolute errors
errors = abs(predictions - test_labels)
 
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
 
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# 


# In[84]:


# 以一个小的随机森林为例，导出其中一个决策树，作图示意

# Limit depth of tree to 2 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features, train_labels)
 
# Extract the small tree
tree_small = rf.estimators_[5]
 
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
 
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
 
graph.write_png('small_tree.png');


# In[83]:


# 变量重要性

# Get numerical feature importances
importances = list(rf.feature_importances_)
 
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
 
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
 
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# list of x locations for plotting

x_values = list(range(len(importances)))

plt.bar(x_values, importances, orientation = 'vertical')

feature_list = ['道路纵坡','曲线半径','曲线长度','路线转角','出口线形', '初始速度','道路编号']
plt.xticks(x_values, feature_list, rotation='30',fontproperties=font_set)

plt.ylabel('重要性',fontproperties=font_set); # plt.xlabel(fontproperties=font_set); plt.title('Variable Importances');
plt.savefig('Importance.png')


# In[40]:


fig,ax = plt.subplots(figsize=(10,5))
ax.plot(test_labels,'-',label='True',linewidth=2)
ax.plot(predictions,'-', label='Predict',linewidth=2)
plt.legend(loc='upper right')
fig.savefig('Predict.png')


# In[71]:


# predictions = np.array(predictions)

pd_TestLabel = pd.DataFrame([predictions,test_labels]).T
pd_TestLabel.columns = ['Predict','True']
pd_TestLabel
pd_TestLabel.to_excel('test_tes.xlsx',index=False)


# In[ ]:




