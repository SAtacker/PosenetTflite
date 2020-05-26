import tensorflow as tf
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

nose = 0
leftEye = 1
rightEye = 2
leftEar = 3
rightEar = 4
leftShoulder = 5
rightShoulder = 6
leftElbow = 7
rightElbow = 8
leftWrist = 9
rightWrist = 10
leftHip = 11
rightHip = 12
leftKnee = 13
rightKnee = 14
leftAnkle = 15
rightAnkle = 16

parser = argparse.ArgumentParser()
args = parser.parse_args()



interpreter = tf.lite.Interpreter(model_path='posenet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
print(f'These are input details : {input_details}')
print(f'These are output details : {output_details}')

cam_width = 1280
cam_height = 720
cam_id = 0

frame_count = 0


cv2.namedWindow("test")
cv2.resizeWindow('test',cam_width,cam_height)
cap = cv2.VideoCapture(cam_id)
cap.set(3, cam_width)
cap.set(4, cam_height)
img_counter = 0



while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    input_data = cv2.resize(frame,(input_details[0]['shape'][1],input_details[0]['shape'][2]),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    input_data = cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB)
    input_data = input_data * (2.0 / 255.0) - 1.0
    # plt.imshow(input_data)
    # plt.pause(0.25)
    input_data = np.expand_dims(input_data, axis=0)
    
    # print(input_data.shape)
    #print(input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    heatmaps_result = np.array(interpreter.get_tensor(output_details[0]['index']))
    offsets_result = np.array(interpreter.get_tensor(output_details[1]['index']))
    displacementFwd_result = np.array(interpreter.get_tensor(output_details[2]['index']))
    displacementBwd_result = np.array(interpreter.get_tensor(output_details[3]['index']))
    heatmaps_result = np.squeeze(heatmaps_result)
    heatmaps_result = np.reshape(heatmaps_result,[9*9,17])
    # y_coords = heatmaps_result // 9
    # x_coords = heatmaps_result % 9
    # print(f'({x_coords},{y_coords})')
    plt.imshow(heatmaps_result)
    plt.pause(0.25)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
plt.show()

cap.release()
cv2.destroyAllWindows()