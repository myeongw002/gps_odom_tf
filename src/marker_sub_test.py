import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class MarkerPathExtractor:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('marker_path_extractor', anonymous=True)
        
        # Subscribe to the global A* path marker topic
        self.path_marker_sub = rospy.Subscriber('/global_a_star_path_marker', Marker, self.path_marker_callback)
        
        # Placeholder to store the extracted path
        self.extracted_path = []
    
    def path_marker_callback(self, msg):
        """
        Callback to process the Marker message and extract path points.
        """
        # Check if the marker type is LINE_STRIP
        if msg.type == Marker.LINE_STRIP:
            extracted_path = []
            
            # Extract points from the marker
            for point in msg.points:
                extracted_path.append([point.x, point.y])  # Only x, y coordinates
            
            # Update the extracted path and print it
            self.extracted_path = np.array(extracted_path)
            rospy.loginfo(f"Extracted path: {self.extracted_path}")
        else:
            rospy.logwarn("Received marker is not of type LINE_STRIP.")
    
    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    extractor = MarkerPathExtractor()
    extractor.spin()
