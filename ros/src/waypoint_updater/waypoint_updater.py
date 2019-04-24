#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.


'''
 
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.pose               = None
        self.base_waypoints     = None
        self.waypoints_2d       = None
        self.waypoint_tree      = None
        self.stopline_wp_idx    = -1
        
        self.loop()
        
    def loop(self):
        rate = rospy.Rate(50) #working in 50Hz
        while not rospy.is_shutdown():
            rospy.loginfo("[YY]looping")
            if self.pose and self.waypoint_tree:
                rospy.loginfo("[YY]contionue")
                closest_waypoint_index = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_index)
            rate.sleep()
       
        
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-2]
        
        # Hyper-plane between two coordinates
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        #If closest way point is behind our car position we go to the next index
        val = np.dot(cl_vect-prev_vect,pos_vect-cl_vect)
        if val > 0:
                closest_idx = (closest_idx +1)%len(self.waypoints_2d)
        return closest_idx
    
    def publish_waypoints(self, closest_idx):
        rospy.logwarn("[YY]calling generate lane")
        final_lane = self.generate_lane(closest_idx)
        self.final_waypoints_pub.publish(final_lane)
        rospy.logwarn("[YY]publish way points")
    
    def generate_lane(self, closest_idx):   
        lane = Lane()
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
       
        if not (self.stopline_wp_idx == -1 or self.stopline_wp_idx >= closest_idx + LOOKAHEAD_WPS):
            lane.waypoints = self.decelerate_waypoints(self.base_waypoints.waypoints[closest_idx:self.stopline_wp_idx], closest_idx)
            
        return lane
    
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # We decrece extra 2 so that the front wheels don't pass the stopline
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2 , 0)
            dist = self.distance(waypoints, 0, stop_idx)
            velocity = math.sqrt(2* MAX_DECEL * dist)
            if velocity < 0.5:
                velocity = 0
            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            if i==stop_idx:
                break
            temp.append(p)
        
        return temp
        
            
    def pose_cb(self, msg):
        self.pose = msg
        rospy.logwarn("[YY]pose_cb")
                
    def waypoints_cb(self, waypoints):
        rospy.logwarn("[YY]waypoints_cb")
        self.base_waypoints = waypoints
        if self.waypoints_2d == None:
            self.waypoints_2d  = [[ waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        
        

    def traffic_cb(self, msg):
        rospy.logwarn("[YY]traffic_cb %d",msg.data)
        self.stopline_wp_idx = msg.data
        

    
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
