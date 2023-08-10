import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from model import load_model,predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
model = load_model()  # Load your pre-trained model

@app.route('/', methods=['GET', 'POST'])
def index():
    
    prediction = []
    if request.method == 'POST':
        print('rec file')
        file = request.files['file']
        
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            values = predict_output(file_path)
            
            

            prediction = predict(model, values)
            print("__")
            print(prediction[0])
            return render_template('index.html', prediction=prediction[0])

    return render_template('index.html', prediction='')
def predict_output(filepath):
    file_path = './static/uploads/sample.txt'
    # Replace with the path of your text file
    values =[]

    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()  # Split the line into words
                for word in words:
                    values.append(float(word[:-1]))
                    
    except FileNotFoundError:
        print("File not found.")


        
    return values


if __name__ == '__main__':
    app.run(debug=True)