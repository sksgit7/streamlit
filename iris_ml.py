import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score 

def main():
	st.title('Iris Flower Classification Web App')    
	st.sidebar.title('Web App sidebar')
	st.markdown("Which flower are you? 🌸")
	#st.sidebar.markdown("Danger ☠️")

	@st.cache
	# to store the function output to disk and use it when the app is re-run
	# rather than executing it everytime (only if the function arguments and
	#										output not change)
	# Also if we call the method more than one time, then at 2nd call it will 
	# use the stored result rather than running the method again
	def load_data():
		data= pd.read_csv('iris.csv')
		return data.drop(['Id'], axis=1)

	@st.cache
	def split(df):
		y=pd.factorize(df.Species)[0]
		#print(pd.factorize(df.Species))
		x=df.drop(columns=['Species'])
		x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3,\
															random_state=0)
		return x_train, x_test, y_train, y_test


	# def plot_metrics():
	# 	st.subheader('Confusion Matrix')
	# 	plot_confusion_matrix(model, x_test, y_test, display_labels= class_names)
	# 	st.pyplot()

		# if 'ROC Curve' in metrics_list:
		# 	st.subheader('ROC Curve')
		# 	plot_roc_curve(model, x_test, y_test)
		# 	st.pyplot()

		# if 'Precision-recall Curve' in metrics_list:
		# 	st.subheader('Precision-recall Curve')
		# 	plot_precision_recall_curve(model, x_test, y_test)
		# 	st.pyplot()

	model_run=False

	df= load_data()
	x_train, x_test, y_train, y_test= split(df)
	#class_names=['Setosa','Virginica','Versicolor']
	class_names=[0,1,2]
	st.sidebar.subheader("Choose Classifier")
	classifier= st.sidebar.selectbox("Classifier",("SVM","LogReg","RForest"))
	if classifier=='SVM':
		st.sidebar.subheader('Hyperparameters')
		C= st.sidebar.number_input("C (Penalty for regularization)",0.01,10.0,\
															step=0.01, key='C')
		gamma= st.sidebar.radio('Gamma (kernel coef)',('scale','auto'), key='gamma')
		kernel= st.sidebar.radio('Kernel',('linear','poly'), key='kernel')

		# metrics= st.sidebar.multiselect("Choose Metric(s)",('Confusion Matrix',\
		# 						'ROC Curve','Precision-recall Curve'))

		if st.sidebar.checkbox('Classify', key='classify'):
			st.subheader('SVM results')
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(x_train,y_train)
			train_accuracy= model.score(x_test,y_test)
			y_pred= model.predict(x_test)
			#st.write('Train accuracy: ', train_accuracy.round(2))
			st.write('Accuracy', accuracy_score(y_test, y_pred).round(2))
			st.write('Precision', precision_score(y_test, y_pred, labels= class_names, average='weighted'))
			st.write('Recall', recall_score(y_test, y_pred, labels= class_names, average='weighted'))
			
			
			st.subheader('Confusion Matrix')
			plot_confusion_matrix(model, x_test, y_test, display_labels= ['Setosa','Versicolor','Virginica'])
			st.pyplot()

			model_run=True

	if classifier=='LogReg':
		st.sidebar.subheader('Hyperparameters')
		C= st.sidebar.number_input("C (Penalty for regularization)",0.01,10.0,\
															step=0.01, key='C_LR')
		max_iter= st.sidebar.slider('Max iterations', 100, 500, key='max_iter')
		# metrics= st.sidebar.multiselect("Choose Metric(s)",('Confusion Matrix',\
		# 						'ROC Curve','Precision-recall Curve'))

		if st.sidebar.checkbox('Classify', key='classify'):
			st.subheader('LogReg results')
			model = LogisticRegression(C=C, max_iter= max_iter)
			model.fit(x_train,y_train)
			train_accuracy= model.score(x_test,y_test)
			y_pred= model.predict(x_test)
			#st.write('Train accuracy: ', train_accuracy.round(2))
			st.write('Accuracy', accuracy_score(y_test, y_pred).round(2))
			st.write('Precision', precision_score(y_test, y_pred, labels= class_names, average='weighted'))
			st.write('Recall', recall_score(y_test, y_pred, labels= class_names, average='weighted'))
			
			st.subheader('Confusion Matrix')
			plot_confusion_matrix(model, x_test, y_test, display_labels= ['Setosa','Versicolor','Virginica'])
			st.pyplot()

			model_run=True

	if classifier=='RForest':
		st.sidebar.subheader('Hyperparameters')
		n_estimators= st.sidebar.number_input("No. of trees", 100, 5000, step=10, key='n_estimators')
		max_depth= st.sidebar.number_input("Max depth", 1, 20, step=1, key='max_depth')
		bootstrap= st.sidebar.radio("Bootstrap samples", (True, False), key='bootstrap')
		# metrics= st.sidebar.multiselect("Choose Metric(s)",('Confusion Matrix',\
		# 						'ROC Curve','Precision-recall Curve'))

		if st.sidebar.checkbox('Classify', key='classify'):
			st.subheader('RForest results')
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth, bootstrap= bootstrap)
			model.fit(x_train,y_train)
			train_accuracy= model.score(x_test,y_test)
			y_pred= model.predict(x_test)
			#st.write('Train accuracy: ', train_accuracy.round(2))
			st.write('Accuracy', accuracy_score(y_test, y_pred).round(2))
			st.write('Precision', precision_score(y_test, y_pred, labels= class_names, average='micro').round(2))
			st.write('Recall', recall_score(y_test, y_pred, labels= class_names, average='micro').round(2))
			
			st.subheader('Confusion Matrix')
			plot_confusion_matrix(model, x_test, y_test, display_labels= ['Setosa','Versicolor','Virginica'])
			st.pyplot()

			model_run=True

	# Checkbox in sidebar to view the dataframe
	if st.sidebar.checkbox("Show raw data", False): # by default unchecked
		st.subheader('Iris dataset')
		st.write(df)

	if model_run==True:
		st.subheader('Predict')
		pl=st.number_input('Petal length') # 3.48  4.48
		pw=st.number_input('Petal width')  # 1.02  1.52
		sl=st.number_input('Sepal length') # 4.98  5.98
		sw=st.number_input('Sepal width')  # 2.01  2.50
		if st.button('Predict'):
			st.write(['Setosa','Versicolor','Virginica'][model.predict([[pl,pw,sl,sw]])[0]])

if __name__ == '__main__':
    main()


