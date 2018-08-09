from keras.preprocessing import image
import numpy as np
from keras.utils.np_utils import to_categorical
import scipy.io as sio
import h5py
from keras.layers import Flatten, Dense,Flatten, Dropout, Reshape, Permute, Activation, Input, merge, TimeDistributed,GlobalMaxPooling1D
from keras.models import Sequential, Model
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Input,Add,Concatenate,Activation
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras.regularizers import l2
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import keras.backend as K
from keras.engine.topology import Layer,InputSpec

from keras.backend.tensorflow_backend import set_session

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.325
sess=set_session(tf.Session(config=config))

class _GlobalTrimmedAveragePool(Layer):
    def __init__(self, **kwargs):
        super(_GlobalTrimmedAveragePool, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)
    def call(self, inputs):
        raise NotImplementedError

class TrimmedAveragePool(_GlobalTrimmedAveragePool):
    def call(self, inputs):
        inputs=tf.transpose(inputs,[0,2,1])
        top= tf.nn.top_k(inputs,k=5)[0]#Dinamic average pooling k=5
        P=tf.reduce_mean(top,axis=-1)
        return P

    

#(input_shape, n_class, num_routing)
def prepareModel():
    """
    Capsule Network on biopsy patches.    
    """
    num_routing=3
    capsule_dimension=8, 
    no_of_channels=16
    x=Input(shape=(224,224,3))
    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', name='conv1')(x)
    conv1 =Activation('relu',name='relu1')(conv1) 
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=5, strides=2, padding='valid') 
    primarycaps = Reshape((53*53,16,8))(primarycaps)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = TimeDistributed(CapsuleLayer(num_capsule=1, dim_capsule=8, num_routing=num_routing,
                             name='digitcaps'))(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)
    output = TrimmedAveragePool()(out_caps)
    model=Model(inputs=x,outputs=output)
    model.summary()
    return model



def executeProg(x_train,y_train,x_test,y_test,nb_epoch,fold):
    split_ratio=.2
    suffledIndex=np.random.permutation(np.shape(x_train)[0])
    length_valid_data=np.int32(np.round(np.size(suffledIndex)*split_ratio))
    x_valid=x_train[suffledIndex[:length_valid_data],:,:,:]
    y_valid=y_train[suffledIndex[:length_valid_data],:]
    
    x_train = np.delete(x_train, suffledIndex[:length_valid_data], axis=0)
    y_train = np.delete(y_train, suffledIndex[:length_valid_data], axis=0)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True
        )
    datagen.fit(x_train)
    
    model = prepareModel();

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(x_train) / batch_size, epochs=nb_epoch,
                              validation_data=[x_valid, y_valid])  
                                                
    history_Path = "./Result/K5Fold{}_Full_History.mat".format(fold+1)
    sio.savemat(history_Path, mdict={'history': history.history});
    model_weight_Path = "./Result/K5Fold{}_Full_Weight.h5".format(fold+1)
    model.save_weights(model_weight_Path)
    
    print("************************ Ended " + " Fold "+str(fold+1)+" ************************");
    Predicted=model.predict(testImg, batch_size=128)
    Acc=model.evaluate(x_test, y_test, batch_size=128)
    acc_Path = "./Result/K5Fold{}_test_acc.mat".format(fold+1)
    sio.savemat(acc_Path, mdict={'Predicted': Predicted,'Acc':Acc});
    print(" Test accuracy= "+str(Acc[1])+"\n")
  
  
#-----------------------Main Program---------------------------------
batch_size = 16
nb_epoch = 100
# .mat file read by h5py.File() possible if the .mat file is saved in v7.3
f = h5py.File('./Data/FinalCroppedDataset.mat','r');
totalImg = f.get('images')
label = f.get('label')
totalImg = np.swapaxes(totalImg, 1, 3)
totalImg=totalImg.astype('float32')
totalImg = totalImg/255.0
label=np.int32(label)
label=2-label
label = np.swapaxes(label, 0, 1)

Size=862; 
for fold in range(1,2):
    testStartRange=Size*fold
    testEndRange=np.min([Size*(fold+1),2585])
    testRange=range(testStartRange,testEndRange)
    trainRange=set(range(2585))-set(testRange)
    trainRange=np.array(list(trainRange))
    #Train Data Preparation
    trainImg=totalImg[trainRange,:,:,:]
    trainLabel=label[trainRange,:]
    #Test Data Preparation
    testImg=totalImg[testRange,:,:,:]
    testLabel=label[testRange,:]
    executeProg(trainImg,trainLabel,testImg,testLabel,nb_epoch,fold)  
