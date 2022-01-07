#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import math
from sklearn.datasets import make_regression
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


#Datasets#
ds16= pd.read_csv('Downloads/raw/Estudiante_5-6 año Secundaria 2016.csv', sep= ',')
dict_2016 = pd.read_excel('Downloads/raw/Diccionarios/Dic_2016_Sec.xlsx')
dict_nvo2 = pd.read_excel('Downloads/dict_nvo.xlsx')

ds16['hab'] = ds16['Ap4'] / ds16['Ap5']
del ds16['Ap4']
del ds16['Ap5']

ds16.shape


# In[2]:


import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
imp2 = SimpleImputer(strategy='mean')
ds16 = ds16.replace('nan','NaN')


# In[3]:


dftc = ds16.drop('Municipio',1)
dftc.lpuntaje = imp2.fit_transform(dftc['lpuntaje'].values.reshape(-1,1))[:,0]


# In[4]:


new_df = imp.fit_transform(dftc)
new_dfp = pd.DataFrame(new_df, columns = dftc.columns)
inner_join = pd.merge(new_dfp, 
                      dict_nvo2, 
                      on ='cod_provincia', 
                      how ='left')


#x = ds16.copy()
#x = pd.merge(x, 
 #           dict_nvo2, 
 ##           on ='cod_provincia', 
#            how ='left')


# In[34]:


#recuento casos x valor
counter = x.apply(pd.Series.value_counts)


# In[5]:


x = inner_join.copy()
bsas = [1]
dstest = (x[x.ponder.isin(bsas)]).infer_objects()
dstest.shape


# In[20]:


x = inner_join.copy()

bsas = ['Buenos Aires']
centro = ['Centro']
nort = ['Norte']
pat = ['Patagonia']
cuyo = ['Cuyo']
dsbsas = (x[x.Region2.isin(bsas)]).infer_objects()
dscent = (x[x.Region2.isin(centro)]).infer_objects()
dsnorte = (x[x.Region2.isin(nort)]).infer_objects()
dspatag = (x[x.Region2.isin(pat)]).infer_objects()
dscuyo = (x[x.Region2.isin(cuyo)]).infer_objects()


# In[6]:


dstest['lpuntaje'].describe()


# In[7]:


df3 = dstest.copy()
delete = pd.read_csv('Downloads/elimina3.csv', sep= ';')
col_list = delete['del']
df3.drop(col_list,axis=True,inplace=True)


# In[8]:


df3['lpuntaje'] = round(df3['lpuntaje'],1)
df3['lpuntaje'].describe()


# In[9]:


df_outlier2 = df3[df3['lpuntaje']< 755].copy()
df_outlier = df_outlier2[df_outlier2['lpuntaje']>275].copy()

print(df3.shape)
print(df_outlier2.shape)
print(df_outlier.shape)


# In[11]:


#Preparo Dataset#
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#df3 = ds16.copy()
#delete = pd.read_csv('Downloads/elimina2.csv', sep= ';')
#col_list = delete['del']
#df3.drop(col_list,axis=True,inplace=True)
#df3.shape
#data = df3.dropna() 
data=df_outlier

#data= ds16
#data = df4
features = [feat for feat in list(data) 
            if feat != 'lpuntaje']

datamat = np.array(features)
X, y = data[features], data.lpuntaje
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.10)

#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestRegressor(n_estimators = 100, random_state=42)
model.fit(X_train, y_train)
print("Accuracy on test data: {:.2f}".format(model.score(X_test, y_test)))
print("Accuracy on train data: {:.2f}".format(model.score(X_train, y_train)))


# In[13]:


import plotly.express as px
df = px.data.tips()
fig = px.box(dstest, x="cod_provincia",y="lpuntaje")
#fig = px.box(df_outlier,y="lpuntaje")
fig.show()


# In[14]:


import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

model2 = sm.OLS(y_train, X_train)
results2 = model2.fit()
#results = sm.OLS(X_test, y_test).predict


# In[15]:


print(results2.summary())


# In[16]:


df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=datamat)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)

#df_feat = df_feature_importances.to_excel('Downloads/Outputs/featimport.xlsx',index= True, header=True)


# In[17]:


# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance del modelo<b>', title_x=0.5)
# The command below can be activated in a standard notebook to display the chart
#fig_features_importance.show()


# We record the name, min, mean and max of the three most important features
slider_1_label = df_feature_importances.index[0]
slider_1_min = int(round(df3[slider_1_label].min()))
slider_1_mean = int(round(df3[slider_1_label].mean()))
slider_1_max = int(round(df3[slider_1_label].max()))

slider_2_label = df_feature_importances.index[1]
slider_2_min = int(round(df3[slider_2_label].min()))
slider_2_mean = int(round(df3[slider_2_label].mean()))
slider_2_max = int(round(df3[slider_2_label].max()))

slider_3_label = df_feature_importances.index[2]
slider_3_min = int(round(df3[slider_3_label].min()))
slider_3_mean = int(round(df3[slider_3_label].mean()))
slider_3_max = int(round(df3[slider_3_label].max()))

slider_4_label = df_feature_importances.index[3]
slider_4_min = int(round(df3[slider_4_label].min()))
slider_4_mean = int(round(df3[slider_4_label].mean()))
slider_4_max = int(round(df3[slider_4_label].max()))

slider_5_label = df_feature_importances.index[4]
slider_5_min = int(round(df3[slider_5_label].min()))
slider_5_mean = int(round(df3[slider_5_label].mean()))
slider_5_max = int(round(df3[slider_5_label].max()))


# In[31]:


app = dash.Dash()

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output

#fuente para definir parametros de layout_ https://codepen.io/chriddyp/pen/bWLwgP.css#

# We apply basic HTML formatting to the layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '1250px', 'font-family': 'Verdana', 'font-size':'10px', 'margin-left':'7%'},
                    children=[

                        # Title display
                        html.H1(children="Herramienta de Simulación - Puntaje en Lengua (Secundaria Arg.)"),
                        
                        # Dash Graph Component calls the fig_features_importance parameters
                        dcc.Graph(figure=fig_features_importance),
                        
                        # The predictin result will be displayed and updated here
                        html.H4(style={'textAlign': 'center', 'font-family': 'Verdana', 'font-size':'15px', 'margin-left':'7%'},
                        id="prediction_result"),
                        
                        # We display the most important feature's name
                        html.H4(children=dict_2016['Variabl_Et'].loc[dict_2016['Variable']==slider_1_label].unique()),

                        # The Dash Slider is built according to Feature #1 ranges

                        dcc.Slider(
                            id='X1_slider',
                            min=slider_1_min,
                            max=slider_1_max,
                            step=1,
                            value=slider_1_mean,
                            marks= dict_2016['Codigo_Et'].loc[dict_2016['Variable']==slider_1_label]
                           ), 

                        # The same logic is applied to the following names / sliders
                        html.H4(children=dict_2016['Variabl_Et'].loc[dict_2016['Variable']==slider_2_label].unique()),

                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=1,
                            value=slider_2_mean,
                            marks= dict_2016['Codigo_Et'].loc[dict_2016['Variable']==slider_2_label]
                        ),

                        html.H4(children=dict_2016['Variabl_Et'].loc[dict_2016['Variable']==slider_3_label].unique()),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=1,
                            value=slider_3_mean,
                            marks= dict_2016['Codigo_Et'].loc[dict_2016['Variable']==slider_3_label]
                        ),

                        html.H4(children=dict_2016['Variabl_Et'].loc[dict_2016['Variable']==slider_4_label].unique()),
                        
                        dcc.Slider(
                            id='X4_slider',
                            min=slider_4_min,
                            max=slider_4_max,
                            step=1,
                            value=slider_4_mean,
                            marks= dict_2016['Codigo_Et'].loc[dict_2016['Variable']==slider_4_label]
                        ),
                        
                        
                        # The predictin result will be displayed and updated here

                        html.H4(children=dict_2016['Variabl_Et'].loc[dict_2016['Variable']==slider_5_label].unique()),
                        
                        dcc.Slider(
                            id='X5_slider',
                            min=slider_5_min,
                            max=slider_5_max,
                            step=1,
                            value=slider_5_mean,
                            marks= dict_2016['Codigo_Et'].loc[dict_2016['Variable']==slider_5_label]
                        ),
                        

                        # The predictin result will be displayed and updated here
                       # html.H4(style={'textAlign': 'center', 'font-family': 'Verdana', 'font-size':'15px', 'margin-left':'7%'},
                        #    id="prediction_result"),

                    ])


# In[32]:


# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"), Input("X4_slider","value"), Input("X5_slider","value")])
# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3, X4, X5):
    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    #input_X = pd.DataFrame(dfnvo.mean(),features).transpose()
    input_X = pd.DataFrame(df_outlier.mean(),features).transpose()
    input_X[slider_1_label] = X1
    input_X[slider_2_label] = X2
    input_X[slider_3_label] = X3
    input_X[slider_4_label] = X4
    input_X[slider_5_label] = X5
    # Prediction is calculated based on the input_X array
    prediction_result = model.predict(input_X)[0]
    # And retuned to the Output of the callback function
    return "Prediction: {}".format(round(prediction_result,1))
if __name__ == "__main__":
    app.run_server()

