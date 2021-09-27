from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
from tensorflow.python.eager.context import num_gpus

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from matplotlib import *
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
tf.device('/gpu:0')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

img_width, img_height =48,48

#Aqui hay que poner los directorios de entrenamiento y validación
train_data_dir ='/Users/theglox/Desktop/Desktop/UD/teleinformatica/tele/tutorial/6 RECONOCIMIENTO FACIAL/data2/train'
validation_data_dir ='/Users/theglox/Desktop/Desktop/UD/teleinformatica/tele/tutorial/6 RECONOCIMIENTO FACIAL/data2/validation'


nb_train_samples = 24176
nb_validation_samples = 5936
epochs = 100
batch_size = 5
kernel_size = 3
pool_size = 2
tamanyo_filtro = 32
num_clases=5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()

#Primera capa oculta de 32 filtros, kernel de 3x3 y poolsize de 2x2
model.add(Conv2D(tamanyo_filtro, (kernel_size,
kernel_size),padding='same', input_shape=input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(pool_size, pool_size),))

#Segunda capa oculta de 64 filtros, kernel de 3x3 y poolsize de 2x2
model.add(Conv2D(2*tamanyo_filtro, (kernel_size,
kernel_size),padding='same')) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

#Tercera capa oculta de 128 filtros, kernel de 3x3 y poolsize de 2x2
model.add(Conv2D(4*tamanyo_filtro, (kernel_size, kernel_size),padding='same')) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten()) 
model.add(Dense(65)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(num_clases)) 
model.add(Activation('sigmoid')) 
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

train_datagen = ImageDataGenerator( rescale=1. / 255,
 shear_range=0.2, 
 zoom_range=0.2, 
 horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory( train_data_dir,target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
							validation_data_dir,
							#color_mode='grayscale',
							target_size=(img_width, img_height),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=False)

history=model.fit_generator(train_generator,steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)
#history=model.fit(train_generator,steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

validation_generator = test_datagen.flow_from_directory(
							validation_data_dir,
							#color_mode='grayscale',
							target_size=(img_width, img_height),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=False)


Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)
print('Matriz de confusión') 
print(confusion_matrix(validation_generator.classes, y_pred))
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0) 
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Pérdidas de entrenamiento', 'Pérdidas de validación'], fontsize=24)
plt.xlabel('Epocas ', fontsize=22)
plt.ylabel('Pérdidas', fontsize=22)

plt.ylim(0,7)
plt.title('Curvas de pérdidas', fontsize=22) 
plt.show()

plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0) 
plt.plot(history.history['val_acc'], 'b', linewidth=3.0) 
plt.legend(['Precisión de entrenamiento', 'Precisión de validación'], fontsize=24)

plt.xlabel('Epocas ', fontsize=22)
plt.ylabel('Precisión', fontsize=22)
plt.ylim(0,1)
plt.title('Curvas de precisión', fontsize=22)
plt.show()