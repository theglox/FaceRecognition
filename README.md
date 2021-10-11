
Dentro de este repositorio se encontraran dos  cosas:
1) el archivo facerecognizer que no se encuentra dentro de ninguna carpeta es la validadcion del documento base
2) La carpeta Facial-Expressions-Recognition cuenta con nuestro programa, el cuela cuenta con la implementacion de el algoritmo haarcascade + el entrenamiento del modelo basado en el documento base. Dentro de esta carpeta encontrara :
  I)Modelo.py = este archivo cuenta con la implementacion de la red neuronal CNN,en ester archivo se hara el entrenamiento con el dataset, las imagenes de validacion y las imagenes de entrenamiento, ademas de esto una vez termine de realizar el entrenamiento el archivo me generara dos graficas las cuales seran las curvas de perdidas y las curvas de precision
  II)haarcascade_frontalface_default.xml es un archivo  que cuenta con el modelo preentrenado de haarcascade proporcionado por opencv, no hace falta modificar nada dentro de este archivo 
  III)Emotion_little_vgg.h5 = este archivo cuenta con los pedos del entrenamiento que va a generar el archivo de la red cnn, el cual se utilizara posteriormente para hacer la prediccion de la emocion a clasificar  
  Iv)Facial_Expressions_Recog.py# FaceRecognition = este archivo importa los dos modelos elhaarcascade_frontalface_default.xml  t Emotion_little_vgg.h5 el cual es  el modelo entrenado por nuestra red neuronal, toma los dos modelos y utliza las clases que se definen dentro de este para realizar una comparacion y poder realizar una prediccion  de la emocion, ya sea dentro de una imagen, una fotografia o un video en vivo.
Aqui se  encuentran las curvas en las pérdidas


![Screenshot](perdidas.png)
Aqui se  encuentran las curvas de precisión


![Screenshot](precision.png)
