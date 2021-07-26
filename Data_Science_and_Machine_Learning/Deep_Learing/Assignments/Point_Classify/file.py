import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dfx=pd.read_csv("Logistic_X_Train.csv")
print(dfx)

X=dfx.values
print(X[0])
print(X.shape)

dfy=pd.read_csv("Logistic_Y_Train.csv")
print(dfy)

Y=dfy.values
Y=Y.reshape((-1,))
print(Y)
print(Y.shape)



dfX_test=pd.read_csv("Logistic_X_Test.csv")
print(dfX_test)

X_Test=dfX_test.values
#print(X_Test)
print(X_Test.shape)


# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=2)
# print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)




model=models.Sequential()
model.add(Dense(3,activation="relu",input_shape=(2,)))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="sigmoid"))



model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


print(model.summary())



# X_val=X_train[:500]
# X_train_new=X_train[500:]

# Y_val=Y_train[:500]
# Y_train_new=Y_train[500:]



X_val=X[:500]
X_train_new=X[500:]

Y_val=Y[:500]
Y_train_new=Y[500:]

print(X_val.shape,Y_val.shape,X_train_new.shape,Y_train_new.shape)


#Starting with epochs =20
hist=model.fit(X_train_new,Y_train_new,epochs=8,batch_size=64,validation_data=(X_val,Y_val)) 

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


print(model.evaluate(X,Y)[1])


Ypred=model.predict(X_Test)
Ypred=Ypred.reshape((-1,))

ans=[]

for i in range(Ypred.shape[0]):
    if Ypred[i]>=0.5:
        ans.append(1)
    else:
        ans.append(0)


pd.DataFrame(ans,columns=['label']).to_csv("ans.csv",index=None)




