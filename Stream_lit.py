#!/usr/bin/env python
# coding: utf-8

# In[32]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[21]:


    
header = st.container()
dataset = st.container()
model_training = st.container()


# In[22]:


with header:
    st.title('Modeling_QC!')
    st.text('In this website Proceeding Quailty Check.')


# In[34]:


with dataset:
    st.header('Golf Modeling')
    
    golf_data = pd.read_csv('st_golf_macro.csv', encoding='cp949')
    st.write(golf_data.head())
    
    st.subheader('Correlation-plot')
    df3 = golf_data.corr()
    mask = np.zeros_like(df3, dtype=np.bool)
    mask[np.triu_indices_from(mask)]= True
    f, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(df3, cmap = 'coolwarm', square = True, mask = mask,
                     vmin = -1, vmax = 1, annot = True, annot_kws = {"size": 10})
    
    st.pyplot(f)

    st.subheader('Pair-plot')
    df2 = golf_data.drop(['few_hole','med_hole','many_hole','수도권','강원권','호서권','호남권','영남권', '제주권'],axis=1)

    labels = ['경과월수', 'log_인구밀집', '업종', 'rating', 'course_len', 'score_seoul','log_pv']
    label_fontdict = {'fontsize' : 14 , 'fontweight':'bold','color':'grey'}
    labelpad = 12

    g = sns.PairGrid(df2)
    g.map_diag(sns.histplot, fill=True)
    g.map_lower(sns.scatterplot)
    g.map_lower(sns.regplot, scatter=False, truncate=False, ci=False)
    g.map_upper(sns.scatterplot)
    g.map_upper(sns.regplot, scatter=False, truncate=False, ci=False)

    g.add_legend()
    for i in range(7):
        g.axes[6,i].set_xlabel(labels[i], fontdict=label_fontdict)
        g.axes[i,0].set_ylabel(labels[i], fontdict=label_fontdict)
    
    st.pyplot(g)

with model_training:
    st.header('Time to train the model!')
    st.text('here you get to choose the features of the model and will see how the performance change')
        
    sel_col, disp_col = st.columns(2)
        
    
    sel_col.text('Here is a list of features in golf_data:')
    sel_col.write(golf_data.columns)
    
    
#    X = golf_data[[input_feature]]
#    y = golf_data[['log_pv']]
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import median_absolute_error as mdape
    from sklearn.metrics import mean_absolute_error as mae
    import statsmodels.formula.api as smf
    
    # 설명변수
    X = golf_data.drop(['log_pv'], axis=1)
    # 반응변수 y
    y = golf_data.log_pv


    logistics_formula='log_pv ~ 경과월수+log_인구밀집+업종+course_len+score_seoul+few_hole+med_hole+many_hole+rating+수도권+강원권+호서권+호남권+영남권+제주권'
    results = smf.ols(logistics_formula, data = golf_data).fit()
    y_pred1 = results.predict(X)
    
#    regr = LinearRegression()
    
#    regr.fit(X, y)
#    prediction = regr.predict(y)
    
    disp_col.subheader('Mean Absolute Error of the model is:')
    disp_col.write(mae(y, y_pred1))
    
    
    disp_col.subheader('Median Absoulte Percentage Error of the model is:')
    disp_col.write(mdape(y, y_pred1))
    
    disp_col.subheader('R_square of the model is:')
    disp_col.write(results.rsquared)
    
    

                                           


