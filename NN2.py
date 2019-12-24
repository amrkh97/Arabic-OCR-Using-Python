import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import keras
import pickle as pkl
import dataset_creator as DC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(X,Y):

    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(Y)
    dummy_y = np_utils.to_categorical(encoded_y,29)

    (trainData,testData,trainLabels,testLabels) = train_test_split(X,Y,test_size=0.01,random_state=42)
    '''
    model = Sequential()
    model.add(Dense(50,input_dim = 17,activation='relu'))
    model.add(Dense(40,activation='relu')) 
    model.add(Dense(29))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
    '''
    model = keras.models.load_model('model_50_40.h5')
    model.fit(trainData,trainLabels,epochs=10,verbose=1)

    (loss,accuracy) = model.evaluate(testData,testLabels,verbose=1)
    print("Loss: {}, accuracy: {}".format(loss,accuracy))
    model.save('model_50_40.h5')


def model_prediction(X):
    D = DC.returnToArabicDictionary()
    model = keras.models.load_model('model_50_40.h5')
    pred = model.predict_classes(X)

    return [D[k] for k in pred]
'''
chunk = pd.read_csv('image_label_pair.csv')
chunk = chunk.values
X = chunk[:,:17].astype(float)
Y = chunk[:,17]
train(X,Y)
'''