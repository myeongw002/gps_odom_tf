#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def read_file_and_publish():
    # Initialize ROS node
    rospy.init_node('textfile_to_rviz_markers', anonymous=True)
    
    # Publisher for markers
    marker_pub = rospy.Publisher('/path_marker', Marker, queue_size=10)
    
    # Load file
    file_path = rospy.get_param("~file_path", "/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path.txt")  # Update this path to your file
    R = rospy.get_param("~R", 0.5)
    points = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.split()
                if len(parts) < 3:
                    continue  # Skip lines with insufficient data
                
                # Extract x, y
                x = float(parts[0])
                y = float(parts[1])
                z = 0.0  # Set Z coordinate to 0 since it's not provided
                
                # Create a Point object
                point = Point()
                point.x = x
                point.y = y
                point.z = z
                
                # Append to points list
                points.append(point)
        rospy.loginfo(len(points))
                
    except IOError as e:
        rospy.logerr("Failed to read file: %s", e)
        return
    
    # Define the Marker message
    marker = Marker()
    marker.header.frame_id = 'map'  # Update to appropriate frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = 'points'
    marker.id = 0
    marker.type = Marker.SPHERE_LIST  # Use SPHERE_LIST to visualize each point as a sphere
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    
    # Set marker scale
    marker.scale.x = R  # Radius of the spheres
    marker.scale.y = R
    marker.scale.z = R
    
    # Set marker color (RGBA)
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0  # Fully opaque
    
    # Add points to marker
    marker.points = points
    
    # Publish the markers in a loop
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        # Update header timestamp
        marker.header.stamp = rospy.Time.now()
        
        # Publish the marker
        marker_pub.publish(marker)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        read_file_and_publish()
    except rospy.ROSInterruptException:
        pass
