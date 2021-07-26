from keras.datasets import imdb
import numpy as np

"""Data Preparation"""

((X_train,Y_train),(X_test,Y_test))=imdb.load_data(num_words=10000)

len(X_train),len(X_test)

len(X_train[0])

print(X_train[0]) #Already conversion of words to numeric data by keras

word_indx=imdb.get_word_index()
print(word_indx.items())

indx_words=dict([val,key] for (key,val) in word_indx.items())
print(indx_words.items())

actual_review=" ".join([indx_words.get(idx-3,"?") for idx in X_train[0]]) # "-3" because the dict adds 3 more elements like pad, s, unk itself
print(actual_review)

"""Vectorize the Data"""

def vectorize_sentences(sentences,dim=10000):
  outputs=np.zeros((len(sentences),dim))

  for i,idx in enumerate(sentences):
    outputs[i,idx]=1
  
  return outputs

X_train=vectorize_sentences(X_train)
X_test=vectorize_sentences(X_test)

X_train.shape,X_test.shape

print(X_train[0])

Y_train=np.array(Y_train,dtype="float32")
Y_test=np.array(Y_test,dtype="float32")

Y_train.shape,Y_test.shape

"""Build The Network



*   Use the fully connected with ReLu Activation
*   2 Hidden Layers with 16 units each
*   1 output layer with 1 unit (Sigmoid Activation)






"""

from keras import models
from keras.layers import Dense

## Define Model

model=models.Sequential()
model.add(Dense(16,activation='relu',input_shape=(10000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

## Compile Model

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

"""Training and Validation"""

X_val=X_train[:5000]
X_train_new=X_train[5000:]

Y_val=Y_train[:5000]
Y_train_new=Y_train[5000:]

X_val.shape,X_train_new.shape,Y_val.shape,Y_train_new.shape

hist=model.fit(X_train_new,Y_train_new,epochs=20,batch_size=512,validation_data=(X_val,Y_val))

print(hist)

"""Visualize"""

import matplotlib.pyplot as plt

h=hist.history

plt.plot(h['val_loss'],label="Validation Loss")
plt.plot(h['loss'],label="Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show() #Here the validation is increasing when training loss decrease => overfitting. We need to stop at that point when train loss is decreased and valid loss is increased.
           # The point is 2.5 epoch

plt.plot(h['val_accuracy'],label="Validation Acc")
plt.plot(h['accuracy'],label="Training Acc")
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.legend()
plt.show() #Here the validation is maxed (88%) at 2.5 epoch. So, we should stop our epoch at 3

model=models.Sequential()
model.add(Dense(16,activation='relu',input_shape=(10000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

## Compile Model

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

hist=model.fit(X_train_new,Y_train_new,epochs=4,batch_size=512,validation_data=(X_val,Y_val))

print(hist)

h=hist.history

plt.plot(h['val_loss'],label="Validation Loss")
plt.plot(h['loss'],label="Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot(h['val_accuracy'],label="Validation Acc")
plt.plot(h['accuracy'],label="Training Acc")
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.legend()
plt.show()

model.evaluate(X_test,Y_test)[1]

model.evaluate(X_train,Y_train)[1]

model.predict(X_test)

