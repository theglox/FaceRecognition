from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
 #tomamos el clasificador haardcascade que viene preentrenado por la libreria de opencv
#face_classifier = cv2.CascadeClassifier('C:\Python37\Projects\Live Project\haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#tomamos el modelo que genreamos gracias a nuestro entrenamiento 
classifier =load_model('/Users/theglox/Desktop/tutorial/6 RECONOCIMIENTO FACIAL/Emotion_little_vgg.h5')
#generamos las clases de las emociones para hacer el labeling correspondiete a las emociones
class_labels = ['Enojado','Feliz','Neutral','triste','Sorpresa']

#en esta parte seleccionamos nuestro metodo de accio, esta parte es la que el usuario va a editar, si quiere  leer ua imagen, hacer la clasificacion en vivo o leer un video que ya se encuentre en el ordenador

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('/Users/theglox/Desktop/documentos/Video.mp4',)
cap = cv2.VideoCapture('/Users/theglox/Desktop/Facial-Expressions-Recognition-master/william.mov',)


while True:
    # aqui vamos a reoger los frames de los videos o iamgenes
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
  #procedemos a hacer la identificacion de las areas de interes con el algoritmo de haarcascade

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # tomamos las areas de interes seleccionadas por el algoritmo y se las pasamos a nuestro entrenamiento para que este haga una prediccion de que clase corresponde esa area de interes

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























