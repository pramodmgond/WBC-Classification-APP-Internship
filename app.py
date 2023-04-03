from flask import Flask, render_template, request # function provided by the flask module
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__) #creates a new Flask app instance.

#loads a pre-trained Keras model 
model = tf.keras.models.load_model('cnn_27.h5')  # load model

#@app.route('/', methods=['GET'])
#def hello_world():
   # return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def predict():
    # get the uploaded image file
    imagefile = request.files.get('imagefile')
    

    
    # check if a file was uploaded
    if not imagefile:
        return render_template('index.html', error="Please select an image file before clicking the predict button.")

    # save the image to a folder on the server
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # preprocess the image
    image = load_img(image_path, target_size=(60, 60))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image/255.0

    # make a prediction using the model
    y_predict = np.argmax(model.predict(image))
    y_predict
    # return the predicted class
    if y_predict==0:
        prediction = "EOSINOPHIL"
    elif y_predict==1:
        prediction = "LYMPHOCYTE"
    elif y_predict==2:
        prediction = "MONOCYTE"
    else:
        prediction = "NEUTROPHIL"
    
    
    return render_template('index.html', prediction=prediction, y_predict=y_predict)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
