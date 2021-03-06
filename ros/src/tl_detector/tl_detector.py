#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier

from scipy.spatial import KDTree

import tf
import cv2
import yaml
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.light_classifier = None
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        
        self.waypoints_2d       = None
        self.waypoint_tree      = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string, Loader=yaml.UnsafeLoader)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        #We initialize classifier last since it takes some time
        self.light_classifier = TLClassifier()
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
       
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if self.waypoints_2d == None:
            self.waypoints_2d  = [[ waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
     
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        #rospy.loginfo("[YY]got image")
        #Need to ensure way points tree is already initialized
        if self.waypoint_tree == None:
            #rospy.logwarn("[YY] not handling image- didn't get base waypoints yet")
            return    
        self.has_image = True
        self.camera_image = msg
        
        light_wp, state = self.process_traffic_lights()
        
        
    
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            #rospy.logwarn("[YY]publish redlight %d",light_wp)
            
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            #rospy.logwarn("[YY]publish redlight %d",light_wp)
        self.state_count += 1

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
      
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Hyper-plane between two coordinates
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect-prev_vect,pos_vect-cl_vect)
        
        if val >0:
                closest_idx = (closest_idx +1)%len(self.waypoints_2d)
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        #If somthing is wrong (no image - or - classifier not up) return RED
        if(not self.has_image or self.light_classifier==None):
            self.prev_light_loc = None
            return TrafficLight.RED

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = -1
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_way_point_idx = self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                d = temp_wp_idx - car_way_point_idx
                if d >= 0 and d <diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        if closest_light:
            state = self.get_light_state(closest_light)
        
        return line_wp_idx, state
                

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
