#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import style
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score


# In[2]:


df = pd.read_csv('Countries Dataset.csv')


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


dfnum = df.drop('Countries', axis=1)
dfnum.head()


# In[7]:


dfnum.corr()


# In[8]:


plt.figure(figsize=(15,9))
sns.heatmap(dfnum.corr(),annot=True,cmap='RdBu')
plt.title('Correlation Heatmap',fontsize=16)
plt.yticks(rotation =0)
plt.show()


# In[9]:


scaler = StandardScaler()
dfstd = scaler.fit_transform(dfnum)


# In[10]:


#Elbow method 

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dfstd)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


# In[11]:


for i in range(3,13):
    labels= KMeans(n_clusters=i,init="k-means++",random_state=0).fit(dfstd).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(silhouette_score(dfstd,labels,metric="euclidean",sample_size=1000,random_state=200)))


# In[12]:


model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11), timings=False)
visualizer.fit(dfstd)
visualizer.show()


# In[13]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)


# In[14]:


kmeans.fit(dfstd)


# In[15]:


dfsegm = dfstd.copy()
dfsegm = pd.DataFrame(data = dfstd,columns = dfnum.columns)
dfsegm['Clusters'] = kmeans.labels_
dfsegm


# In[16]:


pca = PCA()


# In[17]:


pca.fit(dfstd)


# In[18]:


# The attribute shows how much variance is explained by each of the seven individual components.
pca.explained_variance_ratio_


# In[19]:


plt.figure(figsize = (12,9))
plt.plot(range(1,16), pca.explained_variance_ratio_.cumsum(), marker = 'o')
plt.title('Explained Variance by Components',fontsize=16)
plt.xlabel('Number of Components',fontsize=14)
plt.ylabel('Cumulative Explained Variance',fontsize=14)


# In[20]:


pca = PCA(n_components = 2)
pcadf = pca.fit_transform(dfstd)


# In[21]:


wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(pcadf)
    wcss.append(kmeans_pca.inertia_)
    
plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering',fontsize = 16)
plt.show()


# In[22]:


model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11), timings=False)
visualizer.fit(pcadf)
visualizer.show()


# In[23]:


kmeans_pca = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)              


# In[24]:


kmeans_pca.fit(pcadf)


# In[25]:


df_segm_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(pcadf)], axis = 1)
df_segm_pca_kmeans.columns.values[-2: ] = ['Component 1', 'Component 2']

df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

df_segm_pca_kmeans


# In[26]:


df_segm_pca_kmeans['Labels'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'Cluster 3',
                                                          1:'Cluster 1',
                                                          2:'Cluster 2', 
                                                          3:'Cluster 5',
                                                          4:'Cluster 4',                    
                                                                        })


# In[27]:


x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans['Labels'], palette = ['b', 'r', 'g', 'y','m'])
plt.title('Clusters by PCA Components',fontsize=14)
plt.show()
#Final clusters


# In[28]:


Cluster1 = df.loc[df_segm_pca_kmeans['Segment K-means PCA'] == 0]
Cluster1


# In[29]:


Cluster2 = df.loc[df_segm_pca_kmeans['Segment K-means PCA'] == 1]
Cluster2


# In[30]:


Cluster3 = df.loc[df_segm_pca_kmeans['Segment K-means PCA'] == 2]
Cluster3


# In[31]:


Cluster4 = df.loc[df_segm_pca_kmeans['Segment K-means PCA'] == 3]
Cluster4


# In[32]:


Cluster5 = df.loc[df_segm_pca_kmeans['Segment K-means PCA'] == 4]
Cluster5


# In[33]:


c = Cluster4.mean().to_frame()
c = c[0].astype('int')
c = c.to_frame()

c


# In[34]:


for row in c.index:
    print(row, end = " ")


# In[35]:


c.insert(0, "values", ["HTEC", "OLS", "GI", "CCS", "CSEC", "AI", "BDA", "SCM", "IOT", "3D&R", "IDSL", "PA", "GDP", "R&D", "CRM"], True)
c.rename(columns = {0:'numbers'}, inplace = True)
c


# In[36]:


plots = c.plot.bar(figsize=(8, 7), grid= 0, color="#FFEA73")
 
# Iterrating over the bars one-by-one
for bar in plots.patches:
   
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # free space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_height(), 'g'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 25), rotation=90,
                   textcoords='offset points')
plots.spines['top'].set_visible(False)
plots.spines['right'].set_visible(False)
plots.spines['bottom'].set_color('black')
plots.spines['left'].set_color('black')


plt.xticks(rotation=40, horizontalalignment="center", color="black", family = 'monospace', fontsize = "12")

plots.get_legend().remove()
 
# Setting the title for the graph
plt.title("Cluster 5", fontsize="20")
 
# Finally showing the plot
plt.show()


# In[37]:


a = df_segm_pca_kmeans[['Countries', 'CRM']].copy()
a


# In[38]:


plots = a.plot.bar(x='Countries', figsize=(18, 4), grid= 0, color=np.random.rand(3,))
 
# Iterrating over the bars one-by-one
for bar in plots.patches:
   
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # free space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_height(), 'g'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 25), rotation=90,
                   textcoords='offset points')
plots.spines['top'].set_visible(False)
plots.spines['right'].set_visible(False)
plots.spines['bottom'].set_color('black')
plots.spines['left'].set_color('black')


plt.xticks(rotation=70, horizontalalignment="center", color="black", family = 'monospace', fontsize = "12")

plots.get_legend().remove()
 
# Setting the title for the graph

 
# Finally showing the plot
plt.show()


# In[48]:


z = Cluster1.mean().to_frame()
z = z[0].astype('i')
z = z.to_frame().T

z.insert(0, "clusters", ["Cluster3"], True)

z


# In[50]:


y = Cluster2.mean().to_frame()
y = y[0].astype('i')
y = y.to_frame().T

y.insert(0, "clusters", ["Cluster1"], True)

y


# In[51]:


x = Cluster3.mean().to_frame()
x = x[0].astype('i')
x = x.to_frame().T

x.insert(0, "clusters", ["Cluster2"], True)

x


# In[52]:


n = Cluster4.mean().to_frame()
n = n[0].astype('i')
n = n.to_frame().T

n.insert(0, "clusters", ["Cluster5"], True)

n


# In[55]:


m = Cluster5.mean().to_frame()
m = m[0].astype('i')
m = m.to_frame().T

m.insert(0, "clusters", ["Cluster4"], True)

m


# In[57]:


frames = [y, x, z, m, n]
result = pd.concat(frames)

result


# In[144]:


b = result[['clusters', 'CRM']].copy()
b


# In[145]:


plots = b.plot.bar(x='clusters', figsize=(5, 6), grid= 0, color=np.random.rand(3,))
 
# Iterrating over the bars one-by-one
for bar in plots.patches:
   
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # free space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_height(), 'g'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 25), rotation=90,
                   textcoords='offset points')
plots.spines['top'].set_visible(False)
plots.spines['right'].set_visible(False)
plots.spines['bottom'].set_color('black')
plots.spines['left'].set_color('black')


plt.xticks(rotation=30, horizontalalignment="center", color="black", family = 'monospace', fontsize = "12")

plots.get_legend().remove()
 
# Setting the title for the graph
plt.title("CRM", fontsize="15", position=(0,1,0))
plt.xlabel("")
# Finally showing the plot
plt.show()


# In[ ]:




