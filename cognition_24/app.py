from flask import Flask, render_template, request
from V1 import process_image_url  # Import your function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_link = request.form['image_link']
    entity_name = request.form['entity_name']
    prediction = process_image_url(image_link, entity_name)
    return render_template('result.html', image_link=image_link, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
