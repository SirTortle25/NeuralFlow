from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load CIFAR-10 dataset
cifar_dataset = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Preprocess the dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train the model
@app.route('/train_model', methods=['POST',"GET"])
def train_model():
    if request.method=="GET":
        return render_template("image.html")
    else:

        model_architecture = request.form.get('model_architecture')
        model = tf.keras.models.Sequential(eval(model_architecture))

        model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=10)
    # Save the trained model for future use
        model.save('trained_model.h5')
        return 'Model training completed'

# Test the model
@app.route('/test_model', methods=['POST',"GET"])
def test_model():
    if request.method == "GET":
        return render_template()
    else:
        #
        if 'image' in request.files:
            #
            image = request.files['image']
            img = tf.keras.preprocessing.image.load_img(image, target_size=(32, 32))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0

            model = tf.keras.models.load_model('trained_model.h5')
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
        
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            result = class_names[predicted_class]
        
            return 'Test result: ' + result

        return 'No test image found'

if __name__ == '__main__':
    app.run(debug=True)

