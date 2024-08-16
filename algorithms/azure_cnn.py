import tensorflow as tf
import numpy as np
import cv2
from parameters import signs_dict, prediction_treshold

def predict(img) -> str:
    # Prepare your input data
    #img = cv2.imread("traffic signs\\1507.jpg").astype(np.float32)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='models/azure_cnn/model.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run the inference
    interpreter.invoke()

    # Get the output - description of the trafic sign
    output_data = interpreter.get_tensor(output_details[0]['index'])
    indx = np.argmax(output_data[0])
    max = np.max(output_data[0])
    if max > prediction_treshold:
        result = signs_dict[indx]
    else:
        result = None
    return result
