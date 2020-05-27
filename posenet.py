import tensorflow as tf
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

############### This will be done at last ######################
parser = argparse.ArgumentParser()
args = parser.parse_args()
############### Part Ids and their names are stated on Official Google TF Posenet Website ###########
# nose = 0
# leftEye = 1
# rightEye = 2
# leftEar = 3
# rightEar = 4
# leftShoulder = 5
# rightShoulder = 6
# leftElbow = 7
# rightElbow = 8
# leftWrist = 9
# rightWrist = 10
# leftHip = 11
# rightHip = 12
# leftKnee = 13
# rightKnee = 14
# leftAnkle = 15
# rightAnkle = 16
###################### Creating a list of part names #################################
partNames = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
##################### Defining a chain / tree based on nose as root point #############
poseChain = [
    ["nose", "leftEye"], ["leftEye", "leftEar"], ["nose", "rightEye"],
    ["rightEye", "rightEar"], ["nose", "leftShoulder"],
    ["leftShoulder", "leftElbow"], ["leftElbow", "leftWrist"],
    ["leftShoulder", "leftHip"], ["leftHip", "leftKnee"],
    ["leftKnee", "leftAnkle"], ["nose", "rightShoulder"],
    ["rightShoulder", "rightElbow"], ["rightElbow", "rightWrist"],
    ["rightShoulder", "rightHip"], ["rightHip", "rightKnee"],
    ["rightKnee", "rightAnkle"]
]
################# Defining Connected Parts of skeleton ########################
connectedPartNames = [
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
]

######################### Defining Part Ids as a dictionary(parts corresponding to their ids) ###########################3
partIds = {}
for i in range(len(partNames)):
        partIds[partNames[i]] = i
######################## Defining Connected PartIDs corresponding to part names ########################## 

connectedPartIndices = []

def part_indices(connected_part_names, dict_part_ids,connected_part_indices):
    for jointNameA, jointNameB in connected_part_names:
        connected_part_indices.append([dict_part_ids[jointNameA],dict_part_ids[jointNameB]])

part_indices(connectedPartNames, partIds, connectedPartIndices)

# print(connectedPartIndices)
# [[11, 5], [7, 5], [7, 9], [11, 13], [13, 15], [12, 6], [8, 6], [8, 10], [12, 14], [14, 16], [5, 6], [11, 12]]

########################### Defiing Parent Child nodes for displacement gradients ####################
parentChildrenTuples = []

for joint_name in poseChain:
    parent_joint_name = joint_name[0]
    child_joint_name = joint_name[1]
    parentChildrenTuples.append([partIds[parent_joint_name],partIds[child_joint_name]])

######################## Defining the sixteen edges of skeleton ######################
parentToChildEdges = []
for joint_id in parentChildrenTuples:
    parentToChildEdges.append(joint_id[1])

childToParentEdges = []
for joint_id in parentChildrenTuples:
    childToParentEdges.append(joint_id[0])
######################### Further Algorithm ###########################################
# Decode the part positions upwards in the tree, following the backward displacements
# Decode the part positions downwards in the tree, following the forward displacements.
################ Following functions to be added for multipose detection #####################################
# getDisplacement(edgeId, point, displacements)
# StridedIndexNearPoint(point, outputStride, height, width)
# traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId,scoresBuffer, offsets, outputStride, displacements)
# decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd)
# getOffsetPoint(y, x, keypointId, offsets)
# getImageCoords(part, outputStride, offsets)
# clamp(a, min, max)
# squaredDistance(y1, x1, y2, x2)
# ddVectors(a, b)
# clampVector(a, min, max)
################### For single pose ###################################
# get_heatmap_scores
# get_offset_points
# get poses 
# draw skeleton & keypoints
########################################## Tensorflow Lite interpreter #########################
# interpreter = tf.lite.Interpreter(model_path='posenet.tflite')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# ####################### Use help(tf.lite.Interpreter) in case you need some help #####################
# print(f'These are input details : {input_details}')
# print(f'These are output details : {output_details}')
# These are input details : [{'name': 'sub_2', 'index': 93, 'shape': array([  1, 257, 257,   3]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
# These are output details : [{'name': 'MobilenetV1/heatmap_2/BiasAdd', 'index': 87, 'shape': array([ 1,  9,  9, 17]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/offset_2/BiasAdd', 'index': 90, 'shape': array([ 1,  9,  9, 34]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/displacement_fwd_2/BiasAdd', 'index': 84, 'shape': array([ 1,  9,  9, 32]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, {'name': 'MobilenetV1/displacement_bwd_2/BiasAdd', 'index': 81, 'shape': array([ 1,  9,  9, 32]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
############ With the above info we understand that we get heatmaps(scores) , offsets(locations of keypoints) and their displacement gradients/vectors #####################################

################ Defining details of capture window ###################
# cam_width = 1280
# cam_height = 720
# cam_id = 0
# frame_count = 0
# cv2.namedWindow("test")
# cv2.resizeWindow('test',cam_width,cam_height)
# cap = cv2.VideoCapture(cam_id)
# cap.set(3, cam_width)
# cap.set(4, cam_height)
# img_counter = 0
# output_stride = 16


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)
#     # input_data = np.array(tf.image.resize(
#     #                 frame, (input_details[0]['shape'][1],input_details[0]['shape'][2]), method='gaussian', preserve_aspect_ratio=False,
#     #                 antialias=True, name=None
#     #             ))
#     input_data = cv2.resize(frame,(input_details[0]['shape'][1],input_details[0]['shape'][2]),interpolation=cv2.INTER_CUBIC).astype(np.float32)
#     input_data = cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB).astype(np.float32)
#     input_data = input_data * (2.0 / 255.0) - 1.0
#     # plt.imshow(input_data)
#     # plt.pause(0.01)
#     input_data = np.expand_dims(input_data, axis=0)
    
#     # print(input_data.shape)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     heatmaps_result = np.array(interpreter.get_tensor(output_details[0]['index']))
#     offsets_result = np.array(interpreter.get_tensor(output_details[1]['index']))
#     displacementFwd_result = np.array(interpreter.get_tensor(output_details[2]['index']))
#     displacementBwd_result = np.array(interpreter.get_tensor(output_details[3]['index']))
#     heatmaps_result = np.squeeze(heatmaps_result)
#     offsets_result = np.squeeze(offsets_result)
#     displacementFwd_result = np.squeeze(displacementFwd_result)
#     displacementBwd_result = np.squeeze(displacementBwd_result)
#     # print(offsets_result)
#     # plt.imshow()
#     # plt.pause(0.25)
#     # print()
#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
# plt.show()

# cap.release()
# cv2.destroyAllWindows()