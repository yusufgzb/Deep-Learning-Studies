from keras.wrappers.scikit_learn import KerasClassifier#
from sklearn.model_selection import cross_val_score#train test şekilde parçalayıp accury değeri üretiyor
from keras.models import Sequential # yapay sinir Ağ oluşturmak için gerekli
from keras.layers import Dense,Input ,Dropout ,Activation# katmanları inşa etmemizi sağlayan yapı 
import keras
from keras.optimizers import SGD
from sklearn.impute import SimpleImputer

import pandas as pd 
import numpy as np

veri=pd.read_cvs("kanserTespiti.data")
veri.replace('?',-9999,inplace="true")
veriYeni=veri.drop(["1000025"],axis=1)
imp=SimpleImputer(missing_values=-9999,strategy="mean")
veriYeni=imp.fit_transform(veriYeni)

giris=veriYeni[:,0:8]
cikis=veriYeni[:,9]
model=Sequential()
model.add(Dense(64,input_dim=8))
model.add(Activation("relu"))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(32))
model.add(Activation("softmax"))

model.compile(optimizer="adam",loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(giris,cikis,epochs=5,batch_size=32,validation_split=0.13)






