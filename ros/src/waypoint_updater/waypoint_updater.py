#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Useful Variables
        self.pose = None
        self.base_waypoints = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        # Use custom loop rather then rospy.spin() to control publish rate
        self.loop()

    def loop(self):
        # 30 Hz publishing rate
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # Ensure that we have the cars pose and base waypoints before attempting to update them
            if self.pose and self.base_waypoints:
                self.publish_waypoints()
            rate.sleep()

    def waypoints_cb(self, waypoints):
        # Stores waypoints msg in base_waypoints variable which is type Lane
        # This is a latched subscriber meaning that we only recieve the message
        # once since the base_waypoints never change making things more efficient
        self.base_waypoints = waypoints

        # Converting base_waypoints into a KDTree to make look up signficantly
        # more efficient (log(n) v.s. n), also ensure that self.waypoints_2d is
        # initialized before callback
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def get_closest_waypoint_idx(self):
        # Get the cars current x and y position from self.pose msg
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # Query kdtree, first 1 says we only want to return one item, aka closests
        # then you recieve the position[0] and the index[1]
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Grabbing closests and previous coord to check if the closest is ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coord using dotproduct
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        # Positive dotproduct means closest waypoint is behind the car and we
        # modify the closest_idx to be infront of the car
        if val > 0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            # Distance includes number of waypoints back so fron of car stops at the line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # Grabs the pose of the car and stores it
        self.pose = msg

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0

        def dl(a, b): return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. Not currently implemented in program
        pass


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
