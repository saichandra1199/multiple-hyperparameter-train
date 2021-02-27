from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os 
import glob
from scipy import ndimage, misc 
import cv2 
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('-i','--img',type=str,default='',help='path')
ap.add_argument('-d','--destination',type=str,default='',help='path')
ap.add_argument('-n','--number',type=int,default='',help='path')


args=vars(ap.parse_args())

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.5,1.0],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

#ImageDataGenerator(featurewise_center=False, 
# samplewise_center=False, 
# featurewise_std_normalization=False,
#  samplewise_std_normalization=False, 
# zca_whitening=False, zca_epsilon=1e-6,
#  rotation_range=0, width_shift_range=0.,
#  height_shift_range=0., brightness_range=None, 
# shear_range=0., zoom_range=0.,
#  channel_shift_range=0., fill_mode='nearest',
#  cval=0., horizontal_flip=False, vertical_flip=False,
#  rescale=None, preprocessing_function=None,
#  data_format='channels_last', validation_split=0.0,
#  interpolation_order=1, dtype='float32')

images=[]
img= args['img']
destination= args['destination']
n=args['number']
os.mkdir(destination)
#img = ('/home/chandra/Downloads/sample_images2/') # this is a PIL image
#for i in img:
 #   j= cv2.imread(j)
  #  load_img(j)
   # x = img_to_array(i)  # this is a Numpy array with shape (3, 150, 150)
    #x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

for i in os.listdir(img):
        images.append(i)
        # create the full input path and read the file
#print(images)
#print(len(images))
l=[]
for k in images:
                #print(k)
                input_path = os.path.join(img, k)
                #print(input_path)
                j= load_img(input_path)
                x = img_to_array(j)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
                i=0
                for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=destination , save_prefix=destination, save_format='jpg'):
                          i += 1
                          if i > n:
                                  break  # otherwise the generator would loop indefinitely

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

