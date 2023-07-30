import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from model import ANN_Model
model = torch.load('diabetes.pt')
model.eval()
# load the dataset
df = pd.read_csv('diabetes.csv')
# Extract the dependent and independent features
X = df.drop(['Outcome'],axis=1).values
y = df['Outcome'].values

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Load the Standard Scaler for scalling the variables
scaler = pickle.load(open('scaler.pkl','rb'))
X_test = scaler.transform(X_test)
#pickle.dump(scaler,open('scaler.pkl','wb'))
# Creating Tensors
#X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
#y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)



def test_model():
    predictions = []
    with torch.no_grad():
        for i , data in enumerate(X_test):
            y_pred = model(data)
            #print(y_pred.argmax().item())
            y_pred = y_pred.argmax().item()
            predictions.append(y_pred)
    
    cm = confusion_matrix(y_test,predictions)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm,annot=True)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig('confusion_matrix.png',dpi=1200)
    plt.show()


    score = accuracy_score(y_test,predictions)

    print("The Accuracy Score is : {} %".format(score*100))


if __name__ == "__main__":
    test_model()