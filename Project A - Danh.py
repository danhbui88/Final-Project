#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='P:\DATA SCIENCE\Project\Data Project 2/'
filename1= path + 'MtlFire_1.csv'
data=pd.read_csv(filename1, encoding='ISO-8859-1')


# In[2]:


data.head()


# In[3]:


#Change object type to datatimelike type
data['INCIDENT_CreationTime'] = pd.to_datetime(data['INCIDENT_CreationTime'])

#Convert Time column in Main data to Quart so that we can categorize them
data=data.assign(Quart=pd.cut(data.INCIDENT_CreationTime.dt.hour,[-0.01,8,16,24],labels=['jour','soir','nuit']))

#Categorize period of time
dummy=pd.get_dummies(data['Quart'])
df=pd.concat([data,dummy],axis=1)

#Convert the four main types of fire to TRUE, the others are FALSE
d = {'Incendies de bâtiments': True,'AUTREFEU': True, 'INCENDIE': True, 'Sans incendie':False, 'Alarmes-incendies':False, 'Autres incendies':True,
       'Fausses alertes/annulations':False, 'Premier répondant':False,
        '1-REPOND':False, 'SANS FEU':False,
        'FAU-ALER':False, 'NOUVEAU':False}
df['INCIDENT_DescGroup_NEW']=df['INCIDENT_DescGroup'].map(d)

#Convert DESCRIPTION_GROUPE_NEW to category column
df['DESCRIPTION_GROUPE_CAT']=df.INCIDENT_DescGroup_NEW.astype('category')
cat_col=df.select_dtypes(['category']).columns
df[cat_col]=df[cat_col].apply(lambda x: x.cat.codes)
df.head(5)
df=df.drop(['INCIDENT_DescGroup_NEW'],axis=1)


# In[4]:


place_list=['Anjou West', 'Anjou East', "L'ile-Des-Soeurs", 'Verdun North',
            'Verdun South', 'Saint-Léonard North', 'Saint-Léonard West',
            'Saint-Léonard Southeast', 'Saint-Laurent Inner Northeast',
            'Saint-Laurent East', 'Saint-Laurent Outer Northeast',
            'Saint-Laurent Central', 'Saint-Laurent Southwest',
            'Saint-Laurent Southeast', 'Montreal West', 'LaSalle Northwest',
            'LaSalle Southeast', 'Centre-Sud North', 'Centre-Sud South',
            'Plateau Mont-Royal Southeast', 'Old Montreal',
            'Downtown Montreal Northeast', 'Downtown Montreal North',
            'Downtown Montreal East', 'Downtown Montreal Southeast',
            'Downtown Montreal South & West', 'Tour de la Bourse',
            'Place Bonaventure', 'Place Desjardins', 'Petite-Bourgogne',
            'Pointe-Saint-Charles', 'Saint-Henri', 'Ville emard',
            'Plateau Mont-Royal North', 'Plateau Mont-Royal North Central',
            'Plateau Mont-Royal West', 'Outremont',
            'Plateau Mont-Royal South Central',
            'Griffintown (Includes ile Notre-Dame & ile Sainte-Hélène)',
            'Mercier North', 'Mercier West', 'Mercier Southeast',
            'Maisonneuve', 'Hochelaga', 'Ahuntsic North', 'Ahuntsic Central',
            'Ahuntsic East', 'Ahuntsic Southeast', 'Ahuntsic Southwest',
            'Cartierville Northeast', 'Cartierville Central',
            'Cartierville Southwest', 'Rosemont North', 'Rosemont Central',
            'Rosemont South', 'Petite-Patrie Northeast',
            'Petite-Patrie Southwest', 'Saint-Michel West',
            'Saint-Michel East', 'Villeray Northeast', 'Villeray West',
            'Villeray Southeast', 'Parc-Extension', 'Saint-Pierre',
            'Lachine East', 'Lachine West', 
            "L'ile Bizard Northeast",
            "L'ile-Bizard Southwest", 'Sainte-Geneviève',
            'Pointe-Aux-Trembles', 'Rivière-des-Prairies Northeast',
            'Rivière-Des-Prairies Southwest', 'Côte-des-Neiges North',
            'Côte-des-Neiges Northeast', 'Côte-des-Neiges East',
            'Côte-des-Neiges Southwest', 'Notre-Dame-de-Grace Northeast',
            'Notre-Dame-de-Grace Southwest']
df=df[df['PostalCode_PlaceName'].isin(place_list)]


# In[5]:


df1=df.drop(['RecordID', 'postal code', 'Borough_CityNum_Normal','Min_DistKM_Incid_Postal',
             'INCIDENT_CreationDateTime', 'INCIDENT_period','Quart',
             'INCIDENT_DescGroup', 'INCIDENT_TYPE_DESC', 'FIRE INCID / OTHER INCID',
             'INCIDENT_Longitude', 'INCIDENT_Latitude','FireStat_postalCode',
             'FireStat_place', 'LATITUDE_FireStat', 'LONGITUDE_FireStat', 
             'DATE_DEBUT_FireStat', 'DATE_FIN_FireStat',
             'CAT_CRIME_Infractions_entrainant_la_mort_byBorrougCity',
             'CAT_CRIME_Introduction_byBorrougCity',
             'CAT_CRIME_Méfait_byBorrougCity', 
             'CAT_CRIME_Vol_dans___sur_véhicule_à_moteur_byBorrougCity',
             'CAT_CRIME_Vol_de_véhicule_à_moteur_byBorrougCity',
             'CAT_CRIME_Vols_qualifiés_byBorrougCity',
             'CAT_CRIME_total_byBorrougCity', 
             'YConstructionUnknown_ByREM', 
             'Avg_SUPERFICIE_TERRAIN_ByREM', 
             'Avg_ETAGE_HORS_SOL_ByREM', 
             'AvgNumHousingUnPerBuild_ByREM', 
             'AvgBuilSurface_ByREM',
             'YConstruction1600_1915_ByREM', 
             'YConstruction1916_1937_ByREM',
             'YConstruction1938_1952_ByREM', 
             'YConstruction1953_1958_ByREM', 
             'YConstruction1959_1965_ByREM', 
             'YConstruction1966_1978_ByREM', 
             'YConstruction1979_1987_ByREM', 
             'YConstruction1988_1996_ByREM', 
             'YConstruction1997_2005_ByREM',
             'YConstruction2006_2011_ByREM',
             'YConstruction2012_2017_ByREM','Quart'],axis=1)


# In[6]:


#Create Month column from Dates col
df1['Month']=pd.to_datetime(df1['INCIDENT_CreationDate']).dt.month_name().str.slice(stop=3)
#Categorize Month column
dfdummy=pd.get_dummies(df1['Month'],prefix='category')
df1=pd.concat([df1,dfdummy],axis=1)


# In[7]:


df1.head()


# In[8]:


df2=df1[df1.DESCRIPTION_GROUPE_CAT==1]


# In[9]:


df2.shape


# In[10]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(df2['PostalCode_PlaceName'])
le_name_mapping=dict(zip(le.classes_,le.transform(le.classes_)))
print(le_name_mapping)


# In[11]:


df2['PostalCode_PlaceName']=le.fit_transform(df2['PostalCode_PlaceName'])


# In[12]:


df2.head()


# In[13]:


data_2018=df2[df2['INCIDENT_CreationDate'].str.contains("2018")]
data_2019=df2[df2['INCIDENT_CreationDate'].str.contains("2019")]
data_without_2018_2019=df2[~df2['INCIDENT_CreationDate'].str.contains("2018","2019")]


# In[14]:


data_2018.shape,data_2019.shape


# In[15]:


data_without_2018_2019.shape


# In[16]:


data_2018=data_2018.drop(['INCIDENT_CreationDate','INCIDENT_CreationTime','DESCRIPTION_GROUPE_CAT',
                          'Month','NOM_VILLE_ARROND','Borough_City_Normal'],axis=1)
data_2019=data_2019.drop(['INCIDENT_CreationDate','INCIDENT_CreationTime','DESCRIPTION_GROUPE_CAT',
                          'Month','NOM_VILLE_ARROND','Borough_City_Normal'],axis=1)


# In[17]:


data_2018.columns[:28]


# In[18]:


#data_2018[data_2018.columns[0:27]] /= data_2018[data_2018.columns[:27]].max()
data_without_2018_2019=data_without_2018_2019.drop(['INCIDENT_CreationDate','INCIDENT_CreationTime','DESCRIPTION_GROUPE_CAT',
                                          'Month','NOM_VILLE_ARROND','Borough_City_Normal'],axis=1)
#data_without_2018[data_without_2018.columns[0:27]] /= data_without_2018[data_without_2018.columns[0:27]].max()


# In[19]:


data_2018.index=range(0,len(data_2018))
data_2019.index=range(0,len(data_2019))
data_without_2018_2019.index=range(0,len(data_without_2018_2019))


# In[20]:


data_2018.head()


# In[21]:


data_2018.shape,data_2019.shape,data_without_2018_2019.shape


# In[22]:


#Create a function to clean data (missing,NaN,Infinite)
def clean_dataset(df):
  assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
  df.dropna(inplace=True)
  indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
  return df[indices_to_keep].astype(np.float64)


# In[23]:


data_2018=clean_dataset(data_2018)
data_2019=clean_dataset(data_2019)
data_without_2018_2019=clean_dataset(data_without_2018_2019)


# In[24]:


x_train=data_without_2018_2019.drop('PostalCode_PlaceName',axis=1).values
y_train=data_without_2018_2019.PostalCode_PlaceName.values
x_test=data_2018.drop('PostalCode_PlaceName',axis=1).values
x_test1=data_2019.drop('PostalCode_PlaceName',axis=1).values
y_test=data_2018.PostalCode_PlaceName.values
y_test1=data_2019.PostalCode_PlaceName.values


# In[25]:


from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = "gini", random_state = 5)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)


# In[26]:


from sklearn.metrics import accuracy_score
#Predict the response for 2018 dataset
y_pred = clf.predict(x_test)
# Compute accuracy based on validation samples
acc_DT = accuracy_score(y_test, y_pred)
print('Accuracy score',acc_DT)


# In[27]:


#Predict the response for 2019 dataset
y_pred1 = clf.predict(x_test1)
# Compute accuracy based on validation samples
acc_DT1 = accuracy_score(y_test1, y_pred1)
print('Accuracy score',acc_DT1)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[29]:


print(confusion_matrix(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=7)

# Train the model using the training sets
KNN.fit(x_train,y_train)

#Predict Output
predicted= KNN.predict(x_test) # 0:Overcast, 2:Mild
print(predicted)


# In[31]:


y_pred_KNN1 = KNN.predict(x_test)
# Compute accuracy based on validation samples
acc_KNN1 = accuracy_score(y_test, y_pred_KNN1)
print('Accuracy score',acc_KNN1)


# In[78]:


print(confusion_matrix(y_test, y_pred_KNN1))
print(classification_report(y_test, y_pred_KNN1))


# In[32]:


y_pred_KNN2 = KNN.predict(x_test1)
# Compute accuracy based on validation samples
acc_KNN2 = accuracy_score(y_test1, y_pred_KNN2)
print('Accuracy score',acc_KNN2)


# In[79]:


print(confusion_matrix(y_test1, y_pred_KNN2))
print(classification_report(y_test1, y_pred_KNN2))


# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200,max_features=0.25,criterion='entropy')
rf.fit(x_train,y_train)


# In[34]:


#Predict the response for 2018 dataset
y_pred_RF1 = rf.predict(x_test)
# Compute accuracy based on 2018 samples
acc_RF = accuracy_score(y_test, y_pred_RF1)
print('Accuracy score',acc_RF)


# In[80]:


print(confusion_matrix(y_test, y_pred_RF1))
print(classification_report(y_test, y_pred_RF1))


# In[35]:


#Predict the response for 2019 dataset
y_pred_RF2 = rf.predict(x_test1)
# Compute accuracy based on 2019 samples
acc_RF2 = accuracy_score(y_test1, y_pred_RF2)
print('Accuracy score',acc_RF2)


# In[81]:


print(confusion_matrix(y_test1, y_pred_RF2))
print(classification_report(y_test1, y_pred_RF2))


# In[36]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf1 = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf1.fit(x_train, y_train)


# In[37]:


#Predict the response for 2018 dataset
y_pred_svm1 = clf1.predict(x_test)
# Compute accuracy based on 2018 samples
acc_svm = accuracy_score(y_test, y_pred_svm1)
print('Accuracy score',acc_svm)


# In[82]:


print(confusion_matrix(y_test, y_pred_svm1))
print(classification_report(y_test, y_pred_svm1))


# In[38]:


#Predict the response for 2019 dataset
y_pred_svm2 = clf1.predict(x_test1)
# Compute accuracy based on 2019 samples
acc_svm1 = accuracy_score(y_test1, y_pred_svm2)
print('Accuracy score',acc_svm1)


# In[83]:


print(confusion_matrix(y_test1, y_pred_svm2))
print(classification_report(y_test1, y_pred_svm2))


# In[39]:


le_name_mapping1=pd.DataFrame(le_name_mapping,columns=['Place_Name','Code'])


# In[40]:


s=pd.Series(le_name_mapping, name='DataValue')


# In[41]:


s=pd.DataFrame(s)


# In[42]:


s['Place_Name']=s.index


# In[43]:


s.index=range(0,len(s))


# In[44]:


s.columns


# In[45]:


s.head()


# In[46]:


y_pred1.shape


# In[47]:


y_pred1=pd.DataFrame(y_pred1)


# In[48]:


y_pred1.head()


# In[49]:


y_pred1['DataValue']=y_pred1[[0]]


# In[50]:


y_pred1.head()


# In[51]:


table1=pd.merge(s,y_pred1,how='left')


# In[52]:


table1


# In[53]:


result=table1.groupby(by='Place_Name').count()


# In[54]:


result.head()


# In[55]:


result['Place_Name']=result.index
result.index=range(0,len(result))


# In[56]:


result.head()


# In[57]:


result=result.sort_values(by='DataValue',ascending=False)


# In[58]:


result.index=range(0,len(result))


# In[59]:


#result.to_csv(r'P:\DATA SCIENCE\Project\Data Project 2/Pred_result.csv',index = True, header=True, encoding='utf-8')


# In[60]:


result.DataValue.sum()


# In[61]:


#table1=table1[['Place_Name','DataValue']]


# In[62]:


#table1.shape


# In[63]:


#table_2018=df[df['INCIDENT_CreationDate'].str.contains("2018")]


# In[64]:


#table_2018.shape


# In[65]:


#table_2018=table_2018[table_2018.DESCRIPTION_GROUPE_CAT==1]


# In[66]:


#table_2018.reset_index()


# In[67]:


path1='P:\DATA SCIENCE\Project\Data Project 2/'
filename2= path1 + 'Montreal.csv'
data1=pd.read_csv(filename2, encoding='ISO-8859-1')


# In[68]:


data1.head()


# In[69]:


data2=data1[['place name','latitude','longitude']]


# In[70]:


result_predict=pd.merge(result,data2,left_on='Place_Name',right_on='place name',how='left')


# In[71]:


result_predict


# In[72]:


result_predict.to_csv(r'P:\DATA SCIENCE\Project\Data Project 2/Pred_result_coordinates.csv',index = True, header=True, encoding='utf-8')


# In[73]:


list_missing=['Ville emard','Notre-Dame-de-Grace Northeast',"L'ile Bizard Northeast",
                                        "L'ile-Des-Soeurs",'Notre-Dame-de-Grace Southwest','Saint-Pierre','Griffintown (Includes ile Notre-Dame & ile Sainte-Hélène)',
                                        "L'ile-Bizard Southwest",'LaSalle Southeast']


# In[74]:


missing_cor=df[df['PostalCode_PlaceName'].isin(list_missing)]


# In[75]:


missing_cor=missing_cor[['PostalCode_PlaceName','INCIDENT_Longitude', 'INCIDENT_Latitude']]


# In[76]:


missing_cor1=missing_cor[missing_cor.PostalCode_PlaceName=='LaSalle Southeast']


# In[77]:


missing_cor1.describe()


# ## 
