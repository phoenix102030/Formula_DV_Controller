#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from controller.interface import Interface
from controller.controller import Stanley

class Rewind_Stanley(Exception):
    pass

def main():
    interface = Interface()

    #load configuration parameters
    run_mode = rospy.get_param('~run_mode')

    # load car configurations
    max_steering_angle = rospy.get_param('~max_steering_angle')
    min_steering_angle = rospy.get_param('~min_steering_angle')

    # load stanley configurations
    throttle_gain = rospy.get_param('~throttle_gain')
    cross_track_gain = rospy.get_param('~cross_track_gain')
    cross_track_softening = rospy.get_param('~cross_track_softening')
    
    lookahead_distance = rospy.get_param('~lookahead_distance')
    max_heading_error_change = rospy.get_param('~max_heading_error_change')
    # convert from degrees to radians
    max_heading_error_change = max_heading_error_change * np.pi / 180 

    smoothing_factor = rospy.get_param('~smoothing_factor')
    max_steering_change = rospy.get_param('~max_steering_change')

    integral_gain = rospy.get_param('~integral_gain')

    # init stanley controller
    ctl = Stanley(cte_gain=cross_track_gain, 
                  cte_softening=cross_track_softening,
                  lookahead_distance = lookahead_distance,
                  max_heading_error_change = max_heading_error_change,
                  smoothing_factor=smoothing_factor,
                  max_steering_change=max_steering_change,
                  throttle_gain = throttle_gain,
                  int_gain=integral_gain)

    while not interface.is_ready() and not rospy.is_shutdown():
        interface.sleep()


    while not rospy.is_shutdown():
        #fetch data 
        state = interface.inputs['state']
        path = interface.inputs['path']
        path_yaw = interface.inputs['path_yaw']
        speed_profile = interface.inputs['speed_profile']

        # call stanley controller algorithm
        steering_angle, throttle, target_speed = ctl(state, path, path_yaw, speed_profile)     

        """ If the controller outputs steering angle outside of valid bounds, assign max/min value """
        if steering_angle < min_steering_angle:
            steering_angle = min_steering_angle
        elif steering_angle > max_steering_angle:
            steering_angle = max_steering_angle
        
        # publish to vehicle/simulation
        if run_mode == "DV":  
            steering_angle_degrees = (steering_angle * 180 / np.pi)            
            interface.publish_dv_control_target(
                steering_angle=steering_angle_degrees,  
                speed=target_speed
            )
        elif run_mode == "FSDS":
            steering_angle = 2 * (steering_angle - min_steering_angle) / (max_steering_angle - min_steering_angle) - 1
            interface.publish_fsds_control_target(
                steering_angle= -steering_angle,
                throttle=throttle

            )
        interface.sleep()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Rewind_Stanley:
            pass



