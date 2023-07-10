import os
import tempfile
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import expand_dims
import keras
from keras_vggface.utils import preprocess_input

TEMP_DIR=tempfile.gettempdir()
UPLOAD_FOLDER = 'static\img'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_DIR'] = TEMP_DIR

app.config['SECRET_KEY'] = 'your secret key'

image=[{'name': 'name of celebrities',
        'percent': 'percent of similarity',
        'link': 'link to wiki'}]

labels=['Angelina Jolie','Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet',
       'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson',
       'Tom Cruise', 'Tom Hanks', 'Will Smith']

links=['https://ru.wikipedia.org/wiki/Angelina_Jolie', 'https://ru.wikipedia.org/wiki/Brad_Pitt', 'https://ru.wikipedia.org/wiki/Denzel_Washington',
       'https://ru.wikipedia.org/wiki/Hugh_Jackman', 'https://ru.wikipedia.org/wiki/Jennifer_Lawrence', 'https://ru.wikipedia.org/wiki/Johnny_Depp',
       'https://ru.wikipedia.org/wiki/Kate_Winslet', 'https://ru.wikipedia.org/wiki/Leonardo_DiCaprio', 'https://ru.wikipedia.org/wiki/Megan_Fox',
       'https://ru.wikipedia.org/wiki/Natalie_Portman', 'https://ru.wikipedia.org/wiki/Nicole_Kidman', 'https://ru.wikipedia.org/wiki/Robert_Downey_Jr.',
       'https://ru.wikipedia.org/wiki/Sandra_Bullock', 'https://ru.wikipedia.org/wiki/Scarlett_Johansson', 'https://ru.wikipedia.org/wiki/Tom_Cruise',
       'https://ru.wikipedia.org/wiki/Tom_Hanks', 'https://ru.wikipedia.org/wiki/Will_Smith']

model=keras.models.load_model(os.path.join(app.config['UPLOAD_FOLDER'], 'tl_vggface_3'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_face(filename, required_size=(224, 224)):
    img=np.array(Image.open(filename).rotate(-90))
    detector = MTCNN()
    # ectract only face from a photo
    try:
        img=np.array(Image.open(filename))
        results = detector.detect_faces(img)
        x, y, width, height = results[0]['box']
        face = img[y:y+height, x:x+width]
    except:
        img=np.array(Image.open(filename).rotate(-90))
        results = detector.detect_faces(img)
        x, y, width, height = results[0]['box']
        face = img[y:y+height, x:x+width]   
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    return preprocess_input(face_array)
 
 
@app.route('/success', methods=['POST','GET'])
def success():
    return render_template('your_list.html',result=image)

 
 
@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        file=request.files['file']
        if file and allowed_file(file.filename):
            #load file
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['TEMP_DIR'], filename))
            pixels = extract_face(os.path.join(app.config['TEMP_DIR'], filename))

            samples = expand_dims(pixels, axis=0)
            yhat = model.predict(samples)
            # create a list of percents of similarity, celebrities, and links  
            list_1=[]
            for i in range(len(yhat[0])):
                list_1.append((yhat[0][i], labels[i], links[i]))

            def SortFirst(val):
                return val[0]

            list_1.sort(key=SortFirst, reverse=True)
            results=list_1[0:3]
            image.clear()
            for result in results:
                image.append({'name': result[1], 'percent': round(result[0]*100,0), 'link': result[2]})
        return redirect(url_for('success'))
    else: 
        return render_template('index.html')
    
 
 
if __name__ == '__main__':
    app.run(debug=True)
 
 