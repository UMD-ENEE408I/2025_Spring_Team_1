#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

publisherNodeName = "camera_sensor_publisher"
topicName = "video_topic/compressed"

rospy.init_node(publisherNodeName, anonymous=True)
publisher = rospy.Publisher(topicName, CompressedImage, queue_size=1)
rate = rospy.Rate(10) # Reduce FPS to 10 for stability

videoCaptureObject = cv2.VideoCapture(0)
bridgeObject = CvBridge()

while not rospy.is_shutdown():
ret, frame = videoCaptureObject.read()
if ret:
rospy.loginfo("Video frame captured and published")

# Reduce resolution (e.g., 640x480)
frame = cv2.resize(frame, (640, 480))

# Convert to compressed image format
compressedImage = bridgeObject.cv2_to_compressed_imgmsg(frame, dst_format="jpeg")

# Publish compressed image
publisher.publish(compressedImage)

rate.sleep()
