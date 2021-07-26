import os
import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
from keras import models
from keras.layers import Dense



pokemon2label={"Ditto":0,"Gastly":1,"Jynx":2,"Kabuto":3}
label2pokemon={0:"Ditto",1:"Gastly",2:"Jynx",3:"Kabuto"}

p=Path("./dataset")

img_data=[]
label=[]

for dir in p.glob("*"):
    l=str(dir).split("\\")[-1]
    count=0
    for sub_dir in dir.glob("*"):
        sub=str(sub_dir).replace("\\","//")
        img=cv2.imread(sub)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(40,40))
        img_data.append(img)
        label.append(pokemon2label[l])
        count+=1
    
    print(l,count)

img_data=np.array(img_data)
label=np.array(label,dtype="int32")

print(img_data.shape,label.shape)

def draw_img(img,label):
    plt.imshow(img)
    plt.axis("off")
    plt.title(label2pokemon[label])
    plt.show()

for i in range(10):
    r=np.random.randint(210)
    #draw_img(img_data[r],label[r])

img_data=img_data.reshape((img_data.shape[0],-1))
print(img_data.shape)



X_train,X_test,Y_train,Y_test=train_test_split(img_data,label,test_size=0.2,random_state=1)

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)




model=models.Sequential()
model.add(Dense(6,activation="relu",input_shape=(4800,)))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))



model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])


print(model.summary())

X_val=X_train[:30]
X_train_new=X_train[30:]

Y_val=Y_train[:30]
Y_train_new=Y_train[30:]


print(X_train_new.shape,Y_train_new.shape,X_val.shape,Y_val.shape)

hist=model.fit(X_train_new,Y_train_new,epochs=20,batch_size=2,validation_data=(X_val,Y_val))
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
plt.show() #Here the validation is maxed (88%) at 2.5 epoch. So, we should stop our epoch at 3