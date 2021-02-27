# import tensorflow as tf
# from tensorflow import keras
# import os
# from tensorflow import math
# import keras.backend as K
# from tensorflow import math
# import sys
import pyforest
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";

train_data_dir = '/home/chandra/Downloads/new/train'
valid_data_dir = '/home/chandra/Downloads/new/validation'
model_path = "/home/chandra/Downloads/MODELS/Occ_f2_v1_32_1.h5"
#epochs=5
steps_per_epoch =6
validation_steps =20
loss='binary_crossentropy'
img_width = 224
img_height = 224
model =tf.keras.models.load_model(model_path)
string = 0 
epochs = [1,2]
batch_size = [4,8]
learning_rate = [0.0001,0.00001]
for ep in epochs:
    for bs in batch_size:
        for lr in learning_rate:
            print('starting experiments'+ str(string))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

            train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Occlusion','Proper'],
											   class_mode='binary',
											   batch_size=bs,interpolation='lanczos')

            validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['Occlusion','Proper'],
											   class_mode='binary',
											   batch_size=1,interpolation='lanczos')







            model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])

            print('model complied!!')

            print('started training....')
            training = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch,epochs=ep,validation_data=validation_generator,validation_steps=validation_steps)

            print('training finished!!')

            print('saving weights to h5')

            model.save('try' +'_'+ str(string) +'.h5',include_optimizer=False)

            print(lr)
            print(bs)
            print(ep)
            print(string)
            string+=1 

        