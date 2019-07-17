# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:42:02 2019

@author: jjohns
"""

batch_size=128
width=64
height=64
depth=64
nchannel=2    
num_minibatches=len(train1JHN)//batch_size
num_epochs=10
J=[]
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3), activation=tf.nn.relu, input_shape=(height,width,depth,2)))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
model.add(tf.keras.layers.Conv3D(filters=16,kernel_size=(3,3,3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation=tf.nn.relu))
model.compile(optimizer='adam', loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

   
train=train1JHN.sample(n=256,random_state=33)
num_minibatches=len(train)//batch_size
num_epochs=2
J=[]
for epoch_number in range(num_epochs):
    start=time.time()
    for batch_number in range(num_minibatches):
        if epoch_number ==0:
            history,model,X=Learn1JHN_CNN(train.iloc[batch_number*batch_size : (batch_number+1)*batch_size],test1JHN.sample(n=2),model,fname='1JHN'+str(batch_number),file='write')
        else:
            history,model,X=Learn1JHN_CNN(train.iloc[batch_number*batch_size : (batch_number+1)*batch_size],test1JHN.sample(n=2),model,fname='1JHN'+str(batch_number),file='read')
        print('Minibatch # {}'.format(batch_number+1))
        J.append(history.history['mean_absolute_error'])
        if batch_number % 5 ==0:
            plt.plot(J)
            plt.show()
            print('Functional Time to do 5 minibatches is {}'.format(time.time()-start))
    print('Finishing epoch number {}'.format(epoch_number))

#%%
def Learn1JHN_CNN(train1JHN, test1JHN, model,fname, file='write'):
      
    n_folds = 5 #Put me in this block of code

#    def rmsle_cv(model, dataset,y):
#        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dataset)
#        rmse= np.log(-cross_val_score(model, dataset, y, scoring="neg_mean_absolute_error", cv = kf))
#        return(rmse)
    path="X:\\CHAMPS\\"
    if file=='write':
        
        OBConversion=ob.OBConversion()
        OBConversion.SetInFormat("xyz")
        print(len(train1JHN))
        for index in range(0,len(train1JHN)):
            mol=ob.OBMol()
            mol_name=train1JHN.iloc[index]['molecule_name'] +'.xyz'
            OBConversion.ReadFile(mol,mol_name)
            if mol.GetAtom(train1JHN.iloc[index]['atom_index_0'].item()+1).IsNitrogen():
                A=train1JHN.iloc[index]['atom_index_0'].item()+1
                B=train1JHN.iloc[index]['atom_index_1'].item()+1
            else:
                A=train1JHN.iloc[index]['atom_index_1'].item()+1
                B=train1JHN.iloc[index]['atom_index_0'].item()+1
            if index==0:
                X=mi3D.make_conv_input(mol,A,B)
                X=X.reshape((1,64,64,64,2))
                print(X.shape)
            else:
                tmp=mi3D.make_conv_input(mol,A,B)
                X=np.append(X,tmp.reshape(1,64,64,64,2),axis=0)
                
                print(X.shape)
            if index % 32 ==0:
                print('Molecules 1 - {} made into images'.format(index))
        print('index = {}, fname = {}, file = {}, path = {}'.format(index, fname, file, path))
        np.save(path+fname,X)
    else:
        X=np.load(path+fname+".npy")
        print('X loaded successfully')
#    X=(X-np.mean(X))/(np.std(X))
    Y=np.array(train1JHN['scalar_coupling_constant'].reset_index(drop=True))
    Y=Y.reshape((1,len(train1JHN)))
#    print(X.shape)
#    model=keras.Sequential([
#            keras.layers.Dense(256,activation=tf.nn.tanh,input_shape=(64*64*64+6,), kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(128,activation=tf.nn.tanh,kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(16,activation=tf.nn.tanh,kernel_initializer=keras.initializers.he_normal()),
#            keras.layers.Dense(1,activation=tf.nn.relu,kernel_initializer=keras.initializers.he_normal())])
    
    
    history=model.fit(X,Y.T,epochs=1, batch_size=64, verbose=2)
#    plt.plot(history.history['mean_absolute_error'])
#    plt.show()
    
    return history, model,X