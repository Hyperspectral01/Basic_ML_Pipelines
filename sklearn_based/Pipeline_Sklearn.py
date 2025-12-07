import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

df=pd.read_csv('Iris.csv')
df.drop(columns=['Id'],errors='ignore',inplace=True)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

pipeline=Pipeline([
    ('scaler',StandardScaler()),    
    ('classifier',GaussianNB())])

##BY DEFAULT THE METHODS IN THE LAST STEP OF THE PIPELINE WILL BE USED IN THE OBJECT, FOR EARLIER STEPS IN THE PIPELINE, DEFUALT IS fit() or fit_transform()

pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')

with open('pipeline.pkl','wb') as f:
    pickle.dump(pipeline,f)

print("Pipeline saved as 'pipeline.pkl'")

with open('pipeline.pkl','rb') as f:
    loaded_pipeline=pickle.load(f)

sample_data=np.array([[5.1,3.5,1.4,0.2]])
predicted_class=loaded_pipeline.predict(sample_data)                                                                                    

print(f'Predicted class for sample data {sample_data[0]}: {predicted_class[0]}')
