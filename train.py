import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from model import ANN_Model

# load the dataset
df = pd.read_csv('diabetes.csv')
# Extract the dependent and independent features
X = df.drop(['Outcome'],axis=1).values
y = df['Outcome'].values

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Load the Standard Scaler for scalling the variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

pickle.dump(scaler,open('scaler.pkl','wb'))
# Creating Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


def train_model():
    torch.manual_seed(20)
    model = ANN_Model()

    print("Model Parameters : ",model.parameters)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


    epochs = 1000

    final_losses = []

    for i in range(epochs):
        i  = i + 1
        y_pred = model.forward(X_train)
        loss = loss_function(y_pred,y_train)
        final_losses.append(loss)

        if i % 10 == 1:
            print("Epoch number : {} and the loss is : {}".format(i,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_list = [it.item() for it in final_losses]
    plt.plot(range(epochs),loss_list)
    plt.ylabel('Loss')
    plt.xlabel('No. of Epochs')
    plt.savefig('loss_curve',dpi=1200)
    plt.show()

    # save the model
    torch.save(model,'diabetes.pt')



if __name__ == "__main__":
    train_model()