from flask import Flask, render_template, request, url_for
from google.auth.transport import requests
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import json
from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define indices of healthy classes
healthy_indices = [2, 5, 8]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the form
    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")

    # Resize and normalize the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make a prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    words = class_name.split()
    second_word = words[1]

    # Load the JSON file and get the solution for the disease
    with open("package.json") as f:
        data = json.load(f)
    for indices in data["indices"]:
        if indices["id"] == index:
            sol = indices["Solution"]
            break

    # Determine the status of the plant and format the output
    if index in healthy_indices:
        status = "Healthy"
        disease = ""
    else:
        status = "Unhealthy"
        disease = class_name[2:]

    output = {
        "status": status,
        "plant_name": second_word,
        "disease": disease,
        "solution": sol,
        "confidence_score": confidence_score,
    }
    # if output['status'] == 'Unhealthy':
    #     # get the user's email address from the POST request
    #     email_address = request.form['email']
    #
    #     # send email
    #     requests.post(url_for('send_email'), data={'email': email_address})
    #
    #     # show success message
    #     return '''<script>alert("Email sent successfully.")</script>''', 200

    return render_template("index.html", output=output)
@app.route('/send_email', methods=['POST'])
def send_email():
    # get the disease name from the POST request
    # disease_name = request.form['disease_name']

    # get the user's email address from the POST request
    email_address = request.form['email']

    # compose the email message
    subject = 'Plant Disease Detector - More Information Requested'
    message = f'Dear User,\n\nThank you for using Plant Disease Detector. We noticed that your plant was diagnosed with. If you would like to receive more information about this disease and how to treat it, please respond to this email with your questions and comments.\n\nBest regards,\nThe Plant Disease Detector Team'

    # send the email
    try:
        # set up the email message
        msg = MIMEMultipart()
        msg['From'] = 'paul141296@gmail.com'
        msg['To'] = email_address
        msg['Subject'] = subject

        # add the message body
        msg.attach(MIMEText(message, 'plain'))

        # add the image attachment (replace 'path/to/image.jpg' with the actual path to the image file)
        # with open(request.files["image"], 'rb') as f:
        #     img_data = f.read()
        # image = MIMEImage(img_data, name='plant_image.jpg')
        # msg.attach(image)

        # send the email using your Gmail account
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        smtp_username = 'paul141296@gmail.com'
        smtp_password = 'riludzphxhxnjqbt'
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, email_address, msg.as_string())
        server.quit()

        # return a success message
        return jsonify({'success': True, 'message': 'Email sent successfully.'}), 200

    except Exception as e:
        # return an error message
        return jsonify({'success': False, 'message': 'Failed to send email: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
