#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2 # note this is the correct order, swapped from video.
from cv_bridge import CvBridge
from ibvs_controller import *
import time
import numpy as np
from typing import List, Tuple
from geometry_msgs.msg import Twist

print('importing YOLO')
from ultralytics import YOLO
print('done importing YOLO')

subscriberNodeName='camera_sensor_subscriber'
topicName='video_topic/compressed'
bridgeObject=CvBridge()

cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

class IBVS_Controller():
    '''
    An implementation of an image-based visual servoing controller, as described in https://ieeexplore.ieee.org/document/4015997. Developed by Jay Rana (contact: jay.rana1@gmail.com).

    <h2>Quick Start</h2>
    <h3>Camera Frame</h3>
    The camera frame is assumed to be: origin at the center of the image, +x goes to the right, +y goes downwards, and +z goes into the image. All velocities are given relative to this frame (e.g. a positive x velocity means to move right).

    <h3>Point Format</h3>
    Each point is a tuple of three floats.<br>The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0).<br>The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0).<br>The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.

    <h3>Control Modes</h3>
    Several control modes are supported (note: this follows the camera frame noted above):
    <ul>
        <li>Two degrees of freedom: x velocity and z velocity (<code>control_mode='2xz'</code>)</li>
        <li>Two degrees of freedom: z velocity and y angular velocity (<code>control_mode='2zy'</code>)</li>
        <li>Four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity (<code>control_mode='4xyzy'</code>)</li>
    </ul>

    <h3>Interaction Modes</h3>
    Several interaction matrix modes are supported:
    <ul>
        <li>Only use the current positions of each point in the error interaction matrix estimate (<code>interaction_mode='curr'</code>)</li>
        <li>Only use the desired positions of each point in the error interaction matrix estimate (<code>interaction_mode='desired'</code>)</li>
        <li>Use the mean of the error interaction matrix estimates from the current and desired positions (<code>interaction_mode='mean'</code>)</li>
    </ul>
    
    <h3>Controller Loop</h3>
    <ol>
        <li>Instantiate the controller with a control mode (listed above), interaction mode (listed above), and the number of points you will supply to the controller each iteration (must be greater than 0):<br><code>controller = IBVS_Controller(control_mode='2xz', interaction_mode='curr', num_pts=2)</code></li>
        <li>Set the lambda matrix of the controller with a Python list:<br><code>controller.set_lambda_matrix(lambdas=[2.0, 5.0])</code></li>
        <li>Set the desired positions of each of your points in the image:<br><code>controller.set_desired_points(curr_pts=[(-0.5, -0.5, 1.0), (0.5, 0.5, 1.0)])</code></li>
        <li>For each loop of the iteration:</li>
        <ol type="a">
            <li>Set the current positions of each of your points in the image:<br><code>controller.set_current_points(curr_pts=[(-0.2, -0.2, 5.0), (0.2, 0.2, 5.0)])</code></li>
            <li>Check if the error is within some threshold, e.g.:<br><code>if np.linalg.norm(controller.errs) < 0.1:<br>break</code></li>
            <li>Calculate the interaction matrix of the control for this iteration:<br><code>controller.calculate_interaction_matrix()</code></li>
            <li>Calculate the output velocities and save them to a variable:<br><code>vels = controller.calculate_velocities()</code></li>
            <li>Apply the output velocities to your motor controllers (note: your robot may have a different frame than your camera!)
        </ol>
    </ol>

    <h2>Implementation Details</h2>   
    The general control equation is: <code>vels = -1 * lambda_matrix * L_e_est_pinv * errs</code>, where <code>vels</code> is the vector of output velocities, <code>lambda_matrix</code> is the diagonal scaling matrix, <code>L_e_est_pinv</code> is the Moore-Penrose pseudoinverse of the error interaction matrix estimate, and <code>errs</code> is the vector of errors between the current and desired points. <code>vels</code> has dimensions <code>d x 1</code>. <code>lambda_matrix</code> has dimensions <code>d x d</code>. <code>L_e_pinv</code> has dimensions <code>d x 2p</code>. <code>errs</code> has dimensions <code>2p x 1</code>. <code>d</code> denotes the number of degrees of freedom of the controller. <code>p</code> denotes the number of points that will be supplied to the controller.

    To instantiate the controller, call `IBVS_Controller()` with the chosen control mode, interaction mode, and the number of points that will be supplied to the controller. Each point should be a tuple of 3 floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.

    To set the lambda matrix, call `set_lambda_matrix()` with the list of lambda scalars. The list of scalars should have the same length as the number of degrees of freedom of the controller. The list will become a diagonal matrix.

    To calculate `L_e_est_pinv`, call `calculate_interaction_matrix()` after setting the current/desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, `L_e_est_pinv` will be equal to `L_e_pinv`, which is the Moore-Penrose pseudoinverse of the current error interaction matrix. If you are using the `desired` interaction mode, `L_e_est_pinv` will be equal to `L_e_desired_pinv`, which is the Moore-Penrose pseudoinverse of the desired error interaction matrix. If you are using the `mean` interaction mode, `L_e_est_pinv` will be equal to `0.5 * pinv(L_e + L_e_desired)`, where `L_e` is the current error interaction matrix and `L_e_desired` is the desired error interaction matrix.
    
    To set the current positions for each point, call `set_current_points()` with the list of current points. Each point has the same format as in `IBVS_Controller()`. If you are using the `desired` interaction mode, the current depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.
    
    To set the desired positions for each point, call `set_desired_points()` with the list of desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, the desired depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.

    If both the current and desired points have been defined or updated, the controller will automatically calculate the error vector with the `calculate_error_vector()` function.

    Once the lambda matrix, `L_e_est_pinv`, and the error vector have all been set/calculated, you can call `calculate_velocities()` to calculate the output velocities and return them as a NumPy array. The velocities will be in the order listed in the associated control mode.
    '''
    def __init__(self, control_mode: str, interaction_mode: str, num_pts: int):
        assert control_mode == '2xz' or control_mode == '2zy' or control_mode == '4xyzy', f"{control_mode} is not a valid control mode. Please refer to the class docstring to see the list of valid control modes."
        assert interaction_mode == 'curr' or interaction_mode == 'desired' or interaction_mode == 'mean', f"{interaction_mode} is not a valid interaction mode. Please refer to the class docstring to see the list of valid interaction modes."
        assert num_pts > 0, f"{num_pts} is not a valid number of points. You must supply at least one point to the controller each iteration."

        self.control_mode = control_mode

        # two degrees of freedom: x velocity and z velocity
        if self.control_mode == '2xz':
            self.num_degs = 2
        # two degrees of freedom: z velocity and y angular velocity
        elif self.control_mode == '2zy':
            self.num_degs = 2
        # Four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity
        elif self.control_mode == '4xyzy':
            self.num_degs = 4
        
        self.interaction_mode = interaction_mode
        
        self.vels = None
        self.lambda_matrix = None
        self.L_e_est_pinv = None
        self.errs = None
        
        self.num_pts = num_pts
        self.curr_pts = None
        self.desired_pts = None

    def set_lambda_matrix(self, lambdas: List[float]):
        '''
        Given the list of lambda scalars, set `self.lambda_matrix` to a diagonal matrix from that list.

        :param lambdas: A Python list of lambda scalars. Each scalar will scale one component of the final output velocity. The velocities will be in the order listed in the associated control mode.
        '''
        assert len(lambdas) == self.num_degs, f"You must provide {self.num_degs} lambda scalars in the list."
        self.lambda_matrix = np.diag(lambdas)
    
    def calculate_interaction_matrix(self):
        '''
        Calculate the Moore-Penrose pseudoinverse of the error interaction matrix estimate, `self.L_e_est_pinv`, based on our current control and interaction modes.
        '''
        if self.interaction_mode == 'curr':
            assert self.curr_pts is not None, "You must set the current points with set_current_points()."
        elif self.interaction_mode == 'desired':
            assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."
        elif self.interaction_mode == 'mean':
            assert self.curr_pts is not None, "You must set the current points with set_current_points()."
            assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."

        # two degrees of freedom: x velocity and z velocity
        if self.control_mode == '2xz':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(0.0)
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

        # two degrees of freedom: z velocity and y angular velocity
        elif self.control_mode == '2zy':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

        # four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity
        elif self.control_mode == '4xyzy':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 4))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(0.0)
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list2.append(0.0)
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

    def set_current_points(self, curr_pts: List[Tuple[float, float, float]]):
        '''
        Given a set of current points in the image, set `self.curr_pts` to that set of points and calculate the error vector with `self.calculate_error_vector()` if possible.

        :param curr_pts: A Python list of current points, where each point is a tuple of three floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.
        '''
        assert len(curr_pts) == self.num_pts, f"You must provide {self.num_pts} current points."
        self.curr_pts = curr_pts
        if self.desired_pts != None:
            self.calculate_error_vector()

    def set_desired_points(self, desired_pts: List[Tuple[float, float, float]]):
        '''
        Given a set of desired points in the image, set `self.desired_pts` to that set of points and calculate the error vector with `self.calculate_error_vector()` if possible.

        :param desired_pts: A Python list of desired points, where each point is a tuple of three floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.
        '''
        assert len(desired_pts) == self.num_pts, f"You must provide {self.num_pts} desired points."
        self.desired_pts = desired_pts
        if self.curr_pts is not None:
            self.calculate_error_vector()

    def calculate_error_vector(self):
        '''
        This function is automatically called by either `self.set_current_points()` or `self.set_desired_points()` and should not need to be called by the user, even when providing updated current points.
        '''
        assert self.curr_pts is not None, "You must set the current points with set_current_points()."
        assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."
        errors = []
        for i in range(self.num_pts):
            errors.append(self.curr_pts[i][0] - self.desired_pts[i][0])
            errors.append(self.curr_pts[i][1] - self.desired_pts[i][1])
        self.errs = np.reshape(errors, (2 * self.num_pts, 1))

    def calculate_velocities(self) -> np.ndarray:
        '''
        Calculates the output velocities using the general control equation: `vels = -1 * lambda_matrix * L_e_est_pinv * errs`. Ensure that the lambda matrix, current points, and desired points have been set, and that the interaction matrix has been calculated.

        :return: This function returns a NumPy array containing the velocities. The velocities will be in the order listed in the associated control mode.
        '''
        assert self.lambda_matrix is not None, "You must set the lambda matrix with set_lambda_matrix()."
        assert self.L_e_est_pinv is not None, "You must set the error interaction matrix estimate with calculate_interaction_matrix()."
        assert self.errs is not None, "You must set the errors by setting both the current and desired points, or by manually calling calculate_error_vector()."
        self.vels = -1.0 * (self.lambda_matrix @ self.L_e_est_pinv @ self.errs)
        return list(self.vels)

controller = IBVS_Controller(control_mode='2zy', interaction_mode='mean', num_pts=2)
controller.set_lambda_matrix(lambdas=[0.7, 0.7])
controller.set_desired_points(desired_pts=[((205.0 - 320.0)/320.0, (450.0 - 240.0)/240.0, 0.24), ((435.0 - 320.0)/320.0, (450.0 - 240.0)/240.0, 0.24)])

print('importing model')
model = YOLO("yolo-Weights/yolov8n.pt")
print('converting model')
model.to('cuda')
print('done model')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

curr_frame = None
curr_state = "GO_TO_BALL"

def state_go_to_ball():
	global curr_state

	move_cmd = Twist()
	move_cmd.linear.x = 0.0
	move_cmd.angular.z = 0.0
	
	# detect ball and drive to it
	if curr_frame is not None:
		converted_frame = bridgeObject.compressed_imgmsg_to_cv2(curr_frame)
		
		results = model(converted_frame, stream=True)
		
		if len(results) > 0:
			for object in results:
				boxes = object.boxes
				for box in boxes:
					if (classNames[int(box.cls[0])] == "apple" or classNames[int(box.cls[0])] == "frisbee" or classNames[int(box.cls[0])] == "mouse" or classNames[int(box.cls[0])] == "sports ball"):
						x1,y1,x2,y2 = box.xyxy[0]
						y_center = (float(y1) + float(y2))/2.0
						#cv2.rectangle(converted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
						cv2.circle(converted_frame, (int((x1+x2)/2), int((y1+y2)/2)), int((x2-x1)/2), (255, 0, 255), 3)
						
						focal_length = 960.0
						curr_depth = (focal_length * 0.065)/(2 * int((x2-x1)/2))
						#rospy.loginfo(f"curr_depth: {curr_depth}")
						
						curr_pts = [((float(x1) - 320.0)/320.0, (y_center - 240.0)/240.0, curr_depth), ((float(x2) - 320.0)/320.0, (y_center - 240.0)/240.0, curr_depth)]
						#rospy.loginfo(f"pts: {curr_pts}")
						controller.set_current_points(curr_pts=curr_pts)
						controller.calculate_interaction_matrix()
						
						vels = controller.calculate_velocities()
						#rospy.loginfo(f"vels: {vels}")
						move_cmd.linear.x = vels[0][0]
						move_cmd.angular.z = -vels[1][0]
						cmd_vel.publish(move_cmd)
						
						err_norm = np.linalg.norm(controller.errs)
						rospy.loginfo(f"err_norm: {err_norm}")
						if err_norm < 0.5:
							curr_state = "REPOSITION"
		else:
			curr_state = "LOST_BALL"

def state_lost_ball():
	global curr_state
	
	move_cmd = Twist()
	move_cmd.linear.x = 0.0
	move_cmd.angular.z = 0.0
	
	# turn to last known position of ball
	if controller.curr_pts[0][0] < 0:
		move_cmd.angular.z = -0.3
	else:
		move_cmd.angular.z = 0.3
		
	cmd_vel.publish(move_cmd)
	time.sleep(0.1)
	
	curr_state = "GO_TO_BALL"

reposition_timer = 0.0

def state_reposition():
	global curr_state
	global reposition_timer
	
	if 0.0 <= reposition_timer <= 3.0:
		# turn 90 degrees CCW
		move_cmd = Twist()
		move_cmd.linear.x = 0.0
		move_cmd.angular.z = np.pi / 2.0
		cmd_vel.publish(move_cmd)
	elif 3.0 <= reposition_timer <= 4.0:
		# stop
		move_cmd = Twist()
		move_cmd.linear.x = 0.0
		move_cmd.angular.z = 0.0
		cmd_vel.publish(move_cmd)
	elif 4.0 <= reposition_timer <= 10.0:
		# move in a 180 degree arc
		move_cmd = Twist()
		move_cmd.linear.x = 0.3
		move_cmd.angular.z = -np.pi / 4.0
		cmd_vel.publish(move_cmd)
	elif 10.0 <= reposition_timer <= 11.0:
		# stop
		move_cmd = Twist()
		move_cmd.linear.x = 0.0
		move_cmd.angular.z = 0.0
		cmd_vel.publish(move_cmd)
	elif 11.0 <= reposition_timer <= 14.0:
		# turn 90 degrees CW
		move_cmd = Twist()
		move_cmd.linear.x = 0.0
		move_cmd.angular.z = -np.pi / 2.0
		cmd_vel.publish(move_cmd)
	elif 14.0 <= reposition_timer <= 15.0:
		# stop
		move_cmd = Twist()
		move_cmd.linear.x = 0.0
		move_cmd.angular.z = 0.0
		cmd_vel.publish(move_cmd)
	else:
		curr_state = "GO_TO_HUMAN"
		
	reposition_timer += 0.1
	
def state_go_to_human():
	global curr_state
	
	controller.set_desired_points(desired_pts=[((205.0 - 320.0)/320.0, (220.0 - 240.0)/240.0, 0.24), ((435.0 - 320.0)/320.0, (450.0 - 240.0)/240.0, 0.24)])

	move_cmd = Twist()
	move_cmd.linear.x = 0.0
	move_cmd.angular.z = 0.0
	
	# detect ball and drive to it
	if curr_frame is not None:
		converted_frame = bridgeObject.compressed_imgmsg_to_cv2(curr_frame)
		
		results = model(converted_frame, stream=True)
		
		if len(results) > 0:
			for object in results:
				boxes = object.boxes
				for box in boxes:
					if (classNames[int(box.cls[0])] == "scissors"):
						x1,y1,x2,y2 = box.xyxy[0]
						y_center = (float(y1) + float(y2))/2.0
						#cv2.rectangle(converted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
						cv2.circle(converted_frame, (int((x1+x2)/2), int((y1+y2)/2)), int((x2-x1)/2), (255, 0, 255), 3)
						
						focal_length = 960.0
						curr_depth = (focal_length * 0.065)/(2 * int((x2-x1)/2))
						#rospy.loginfo(f"curr_depth: {curr_depth}")
						
						curr_pts = [((float(x1) - 320.0)/320.0, (float(y1) - 240.0)/240.0, curr_depth), ((float(x2) - 320.0)/320.0, (float(y2) - 240.0)/240.0, curr_depth)]
						#rospy.loginfo(f"pts: {curr_pts}")
						controller.set_current_points(curr_pts=curr_pts)
						controller.calculate_interaction_matrix()
						
						vels = controller.calculate_velocities()
						#rospy.loginfo(f"vels: {vels}")
						move_cmd.linear.x = vels[0][0]
						move_cmd.angular.z = -vels[1][0]
						cmd_vel.publish(move_cmd)
						
						err_norm = np.linalg.norm(controller.errs)
						rospy.loginfo(f"err_norm: {err_norm}")
						if err_norm < 0.1:
							curr_state = "STOP"
		else:
			curr_state = "LOST_HUMAN"

def state_lost_human():
	global curr_state
	
	move_cmd = Twist()
	move_cmd.linear.x = 0.0
	move_cmd.angular.z = 0.0
	
	# turn to last known position of ball
	if controller.curr_pts[0][0] < 0:
		move_cmd.angular.z = -0.3
	else:
		move_cmd.angular.z = 0.3
		
	cmd_vel.publish(move_cmd)
	time.sleep(0.1)
	
	curr_state = "GO_TO_HUMAN"

def displayCallback(event):
	#rospy.loginfo('displaying frame')
	
	move_cmd = Twist()
	move_cmd.linear.x = 0.0
	move_cmd.angular.z = 0.0
	
	if curr_frame is not None:
		converted_frame = bridgeObject.compressed_imgmsg_to_cv2(curr_frame)
		
		rospy.loginfo(f"CURRENT STATE: {curr_state}")
	
		if curr_state == "GO_TO_BALL":
			state_go_to_ball()
		elif curr_state == "LOST_BALL":
			state_lost_ball()
		elif curr_state == "REPOSITION":
			state_reposition()
		elif curr_state == "GO_TO_HUMAN":
			state_go_to_human()
		elif curr_state == "LOST_HUMAN":
			state_lost_human()
		else: # STOP or invalid state
			cmd_vel.publish(move_cmd)
		
		cv2.imshow("camera", converted_frame)
		cv2.waitKey(1)
	
	'''
	mask = cv2.inRange(converted_frame, (70, 0, 0), (255, 60, 60))
	mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	converted_frame = converted_frame & mask_rgb

	gray = cv2.cvtColor(converted_frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 5)
	#_, gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0], param1=255, param2=10, minRadius=1, maxRadius=200)

	if circles is not None and len(circles) <= 1:
		circles = np.uint16(np.around(circles))
		for i in circles[0, :]:
			center = (float(i[0]), float(i[1]))
			#rospy.loginfo(f"center: {center}")
			cv2.circle(converted_frame, (int(center[0]), int(center[1])), 1, (0, 100, 100), 3)
			radius = float(i[2])
			#rospy.loginfo(f"radius: {radius}")
			cv2.circle(converted_frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 255), 3)
			
			#print(f"math: {center[0]} {center[0] - radius} {center[0] - radius - 320.0} {(center[0] - radius - 320.0)/320.0}")
			
			focal_length = 960.0
			curr_depth = (focal_length * 0.065)/(2 * radius)
			#rospy.loginfo(f"curr_depth: {curr_depth}")
			
			curr_pts = [((center[0] - radius - 320.0)/320.0, (center[1] - 240.0)/240.0, curr_depth), ((center[0] + radius - 320.0)/320.0, (center[1] - 240.0)/240.0, curr_depth)]
			#rospy.loginfo(f"pts: {curr_pts}")
			controller.set_current_points(curr_pts=curr_pts)
			controller.calculate_interaction_matrix()
			
			vels = controller.calculate_velocities()
			#rospy.loginfo(f"vels: {vels}")
			move_cmd.linear.x = vels[0][0]
			move_cmd.angular.z = -vels[1][0]
	'''


def callbackFunction(message):
	#bridgeObject = CvBridge()

	#rospy.loginfo("received a video message/frame")

	#convertedFrameBackToCV = bridgeObject.imgmsg_to_cv2(message)

	#cv2.imshow("camera", convertedFrameBackToCV)

	#cv2.waitKey(1)

	global curr_frame
	#curr_frame = convertedFrameBackToCV
	curr_frame = message
	#rospy.loginfo("curr_frame: " + str(curr_frame))


rospy.init_node(subscriberNodeName, anonymous=True)
rate = rospy.Rate(10)
rospy.Subscriber(topicName, CompressedImage, callbackFunction, queue_size=1, buff_size=2**32)
rospy.Timer(rospy.Duration(1.0/10.0), displayCallback)
rospy.spin()
#while not rospy.is_shutdown():
	#if curr_frame is not None:
		#cv2.imshow("camera", curr_frame)
		#cv2.waitKey(1000)
	#rate.sleep()
cv2.destroyAllWindows()



