#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge

subscriberNodeName = "camera_sensor_subscriber"
topicName = "video_topic/compressed"

rospy.loginfo("Here")

def callbackFunction(message):
	bridgeObject = CvBridge()
	rospy.loginfo("Received a video frame")

	# Convert from compressed image
	frame = bridgeObject.compressed_imgmsg_to_cv2(message)

	cv2.imshow("camera", frame)
	cv2.waitKey(1)

rospy.init_node(subscriberNodeName, anonymous=True)
rospy.Subscriber(topicName, CompressedImage, callbackFunction, queue_size=1)
rospy.spin()
cv2.destroyAllWindows()
