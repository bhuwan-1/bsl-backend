from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from words import sign_dictionary, sign_classes
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

model_path = "GRU_200_1000_256_1.h5"
model = tf.keras.models.load_model(model_path)

@app.route('/api/signs/<word>')
def get_signs(word):
    if word in sign_dictionary:
        return jsonify(sign_dictionary[word])
    else:
        return jsonify({"meaning": "Not found"})
    
@app.route('/api/words')
def get_words():
    return jsonify(sign_classes)

sequence = []
sequence_length = 30
prediction_threshold = 0.6

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('keypoints')
def handle_keypoints(data):
    # Convert data to appropriate numpy format if necessary
    keypoints = np.array(data)
    global sequence
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    if len(sequence) == sequence_length:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        if np.max(res) > prediction_threshold:
            predicted_class = np.argmax(res)
            print("Predicted class: ", predicted_class)
            eng_word, dzongkha_word = sign_classes[predicted_class]
                
            emit('data', {'english_word': eng_word, 'dzongkha_word': dzongkha_word, })


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
