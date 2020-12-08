
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.title('IRIS EDA')

#flowers
@st.cache
def image(img):
	flower=Image.open(img)
	return flower

species= st.selectbox('Choose species',('Setosa',\
	                             'Versicolor',\
	                                         'Virginica'))
if species != None:
	st.image(image(species+'.jpg'), caption=species)

#load dataset
@st.cache
def data():
	df=pd.read_csv('iris.csv')
	return df

#checkbox to show data
if st.checkbox('View dataset'):
	if st.button('Full'):
		st.write(data())
	elif st.button('Head'):
		st.write(data().head())

data=data().drop(['Id'],axis=1)
#info about dimension
dim= st.radio('Rows or Cols?',('Rows','Cols'))
if dim=='Rows':
	st.write('Number of rows:',data.shape[0])
else:
	st.write('Number of cols:',data.shape[1])

# Stats
if st.checkbox('Description/Stats'):
	st.write(data.describe())
	st.dataframe(data.describe())
	st.table(data.describe())

#Plots
#Distribution of features
dist= st.selectbox('Choose a feature',('None',\
				'petalLength','petalWidth',\
				'sepalLength','sepalWidth'))
if dist=='None':
	pass
elif dist=='petalLength':
	fig, ax = plt.subplots()
	ax.hist(data.PetalLengthCm)
	st.pyplot(fig)
	# st.write(data.PetalLengthCm.hist())
	# st.pyplot()
elif dist=='petalWidth':
	st.write(data.PetalWidthCm.hist())
	st.pyplot()
elif dist=='sepalLength':
	st.write(data.SepalLengthCm.hist())
	st.pyplot()
else:
	st.write(data.SepalWidthCm.hist())
	st.pyplot()

#Bivariate analysis
features= st.multiselect('Select two features:',\
	('PetalLength',\
				'PetalWidth','SepalLength','SepalWidth'))
if len(features)!=2:
	st.error('Choose two features')
else:
	fig, ax = plt.subplots()
	ax=sns.scatterplot(x=features[0]+'Cm', y=features[1]+'Cm',\
						data=data, hue='Species')
	st.pyplot(fig)

#Correlation matrix
if st.checkbox('Correlation matrix'):
	corr= st.radio('Matplotlib or Seaborn',('Matplotlib','Seaborn'))
	fig, ax= plt.subplots(figsize=(7,6))
	if corr=='Matplotlib':
		ax.matshow(data.corr())
		st.pyplot(fig)
	else:
		ax=sns.heatmap(ax=ax,data=data.corr())
		st.pyplot(fig= fig) # we can pass this parameter also
		
#Bar plot
if st.checkbox('Show bar plot'):
	group= data.groupby('Species').agg('mean')
	st.bar_chart(group)

#Area chart
if st.checkbox('Show area chart'):
	species=st.selectbox('Species',('Iris-setosa', 'Iris-versicolor',\
										'Iris-virginica'))
	group= data[data.Species==species].iloc[:,0:4]
	st.area_chart(group)


# import time
# for x in range(60):
# 	st.balloons()
# 	time.sleep(1)
	