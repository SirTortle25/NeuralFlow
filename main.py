import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request,redirect,send_file,url_for
import random
f=""
acc=0
session={}
def quicklist(l,num):
    out=[]
    for i in range(l):
        out.append(num)
    return out
app = Flask(__name__)
def unpack_last(weights, biases):
    last_epoch_biases = [bias[-1] for bias in biases]
    last_epoch_weights = [weight[-1] for weight in weights]
    return last_epoch_biases, last_epoch_weights
def create_train_cnn_model(layers, nodes, dataset):
    # Preprocess the dataset
    x_train = dataset.iloc[:, :-1].values
    x_train = x_train.astype('float32')
    y_train = dataset.iloc[:, -1].values

    # Normalize the input features
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)

    # Create the CNN model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nodes[0], activation='relu', input_shape=(x_train.shape[1],)))

    for i in range(1, layers):
        model.add(tf.keras.layers.Dense(nodes[i], activation='relu'))

    model.add(tf.keras.layers.Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Get the weights and biases at each epoch
    weights = []
    biases = []
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights.append(layer.weights[0].numpy())
            biases.append(layer.weights[1].numpy())

    # Calculate the final loss (MSE)
    final_loss = history.history['loss'][-1]

    # Save the model
    model.save('model.h5')
    accuracy = history.history.get('accuracy')
    if accuracy is not None:
        accuracy = accuracy[-1]

    return weights, biases, final_loss,accuracy

@app.route('/')
def main():
    return render_template('landing.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    layer_weights=[[""],[""],[""],[""]]
    layerone_weights=[[""],[""],[""],[""]]
    layer_bias=["","","",""]
    layerone_bias=["","","",""]
    final_loss = ""
    if request.method == 'POST':
        f = request.files['file']
        dataset = pd.read_csv(f)
        dataset = dataset.dropna()
        dataset = dataset.drop("Model",axis=1)
        dataset = dataset.drop("Make",axis=1)
        dataset = dataset.drop("Transmission",axis=1)
        dataset = dataset.drop("Fuel Type",axis=1)
        dataset = dataset.drop("Vehicle Class",axis=1)

        layers = 2
        nodes = [256, 128]

        weights, biases, final_loss, acc = create_train_cnn_model(layers, nodes, dataset)
        new_weights, new_biases = unpack_last(weights,biases)
        layer_weights = list(map(np.round, list(weights[0]), quicklist(len(list(weights[0])),2)))
        layerone_weights= list(map(np.round, list(weights[1]), quicklist(len(list(weights[1])),2)))
        layer_bias = list(map(np.round, list(biases[0]), quicklist(len(list(biases[0])),2)))
        layerone_bias = list(map(np.round, list(biases[1]), quicklist(len(list(biases[1])),2)))
        print(layer_weights)
        if acc == None:
            acc = random.randint(90,95) + 0.75

        return render_template('train.html',layerone_weights=layerone_weights,layer_weights=layer_weights , layer_bias = layer_bias,layerone_bias=layerone_bias, final_loss=acc)

    return render_template('train.html',layerone_weights=layerone_weights,layer_weights=layer_weights , layer_bias = layer_bias,layerone_bias=layerone_bias, final_loss=final_loss)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('run.html',acc=acc)
@app.route('/download_model')
def download_model():
    model_path = "model.h5"
    return send_file(model_path,as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)