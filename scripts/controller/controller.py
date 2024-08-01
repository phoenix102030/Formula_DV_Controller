import numpy as np
import rospy
# -*- coding: utf-8 -*-

class Stanley:
    def __init__(self, 
                 cte_gain,
                 cte_softening,
                 lookahead_distance,
                 max_heading_error_change,
                 smoothing_factor,
                 max_steering_change,
                 throttle_gain,
                 int_gain
        ):
        """setup stanley"""
        # Cross track error parameters
        self.cte_gain = cte_gain
        self.cte_softening = cte_softening

        # Heading error parameters
        self.lookahead_distance = lookahead_distance
        self.max_heading_error_change = max_heading_error_change

        # Steering angle smoothing parameters
        self.smoothing_factor = smoothing_factor
        self.max_steering_change = max_steering_change
        
        # PI parameters
        self.throttle_gain = throttle_gain
        self.integral_gain = int_gain

        # "Run time variables"
        self.throttle = 0.
        self.integral_error = 0.
        self.steering_angle = 0.

        self.vehicle_position = None
        self.vehicle_yaw = None
        self.vehicle_velocity = None

        self.path = None
        self.path_yaw = None
        self.speed_profile = None

        self.nearest_point = None
        self.nearest_point_index = None
        self.lookahead_point = None
        self.lookahead_point_index = None

        self.target_speed = None

    def __call__(self, 
                 state,
                 path,
                 path_yaw,
                 speed_profile
        ):
        """
        Execution method
        """
        self.vehicle_position = [state[0], state[1]]
        self.vehicle_yaw = state[2]
        self.vehicle_velocity = state[3]
        self.path = path
        self.path_yaw = path_yaw

        self.speed_profile = speed_profile
        self.nearest_point, self.nearest_point_index = self.find_nearest_point()
        
        self.lookahead_point, self.lookahead_point_index = self.find_lookahead_point()
        
        self.throttle, self.target_speed = self.proportional_control()
        self.steering_angle = self.stanley_control()

        steering_angle = self.steering_angle
        throttle, target_speed = self.throttle, self.target_speed
        return steering_angle, throttle, target_speed
    
    def proportional_control(self): 
        """
        Proportional and integral parts of PID controller. 
        """
        # PI CONTROLLER
        target_speed = self.speed_profile[self.lookahead_point_index]
        error = target_speed - self.vehicle_velocity
        # Proportional term
        p = self.throttle_gain * error
        # Integral term with anti-windup
        if -1 < self.throttle < 1:  # Only accumulate error if throttle is not saturated
            self.integral_error -= error
            self.integral_error = max(min(self.integral_error, 10), -10)
        i = self.integral_gain * self.integral_error
        # Calculate total throttle adjustment
        throttle = p + i
        throttle = min(max(throttle, -1), 1)

        return throttle, target_speed
    

    def stanley_control(self):
        """
        Stanley lateral controller.
        Computes steering angle as sum of cross track and heading error angles
        """
        # Stearing anlge is computed as sum of heading error and cross track error
        heading_error_angle = 0
        cross_track_error_angle = 0

        heading_error_angle += self.calculate_heading_error_angle()
        cross_track_error_angle += self.calculate_cross_track_error_angle()

        steering_angle = heading_error_angle + cross_track_error_angle
        
        # Steering anlge is smoothed out
        prev_steering_angle = getattr(self, 'prev_steering_angle', 0)
        smoothing_factor = self.smoothing_factor
        
        desired_steering_angle = (1 - smoothing_factor) * prev_steering_angle + smoothing_factor * steering_angle
        
        max_steering_change_per_update = self.max_steering_change
        steering_change = desired_steering_angle - prev_steering_angle

        steering_change_degrees = steering_change * 180 / np.pi

        if abs(steering_change_degrees) > max_steering_change_per_update:
            max_steering_change_per_update_rads = max_steering_change_per_update * np.pi / 180
            steering_change = np.sign(steering_change) * max_steering_change_per_update_rads

        steering_angle = prev_steering_angle + steering_change

        self.prev_steering_angle = steering_angle

        return steering_angle
    
    def calculate_cross_track_error_angle(self):
        """
        Computes crosstrack error by meassuring distance between vehicle and 
        closest point (lookahead point) on track. Assigns negative or positive 
        crosstrack error depending on if vehicle is to the left or right of 
        planned path.  
        """
        cte = np.linalg.norm(np.array(self.vehicle_position) - np.array(self.nearest_point))
        arctan_arg = self.cte_gain * cte / (self.vehicle_velocity + self.cte_softening)
        cte_angle = np.arctan(arctan_arg)
        
        # Determine if vehicle is left/right of planned path
        prev_point = np.array(self.path[self.nearest_point_index - 10])
        vehicle_vec =  np.append(self.vehicle_position - prev_point, 0)            # 3d-vector from vehicle's second nearest path point on track to vehicle's position
        path_vec = np.append(self.nearest_point - prev_point, 0)                   # 3d-vector from vehicle's second nearest path point to vehicle's nearest path point
        dir_vec = np.cross(path_vec, vehicle_vec)                                  # positive cross product if vehicle left of track, negative cross product if vehicle right of track

        if dir_vec[2] >= 0:
            return -cte_angle
        else:
            return cte_angle



    def calculate_heading_error_angle(self):
        """
        Calculate the heading error between the vehicle's heading and the path direction.
        """
        self.path_lookahead_yaw = self.path_yaw[(self.lookahead_point_index) % len(self.path_yaw)]
        heading_error_angle = self.path_lookahead_yaw - self.vehicle_yaw

        # Normalize if heading error is off by 2 pi
        if abs(heading_error_angle) > np.pi:
            if abs(heading_error_angle + 2*np.pi) < np.pi:
                heading_error_angle += 2*np.pi
            elif abs(heading_error_angle - 2*np.pi) <np.pi:
                heading_error_angle -= 2*np.pi

        # Heading error filter, s.t heading error cant change by more than max_heading_change per time step
        prev_heading_error = getattr(self, 'prev_heading_error', heading_error_angle)
        if abs(prev_heading_error - heading_error_angle) > self.max_heading_error_change:
            if prev_heading_error - heading_error_angle < 0:
                heading_error_angle = prev_heading_error + self.max_heading_error_change
            else:
                heading_error_angle = prev_heading_error - self.max_heading_error_change

        self.prev_heading_error = heading_error_angle

        return heading_error_angle

    def find_nearest_point(self):
        """
        Find the nearest point on the path to the current position.
        path: List of points [(x1, y1), (x2, y2), ...]
        position: Current position (x, y)
        """
        path = np.array(self.path)
        vehicle_position = np.array(self.vehicle_position)
        distances = np.sqrt(((path - vehicle_position) ** 2).sum(axis=1))
        nearest_index = distances.argmin()
        return path[nearest_index], nearest_index


    def find_lookahead_point(self):
        """
        Finds the path point [lookahead_distance] meters in front of nearest point.
        """
        start_point = self.nearest_point
        ind = (self.nearest_point_index + 1) % len(self.path)

        while True:
            ind = (ind + 1) % len(self.path)
            if ind == self.nearest_point_index:
                rospy.logward('[Stanley] No lookahead_point found')
                return None, None
            distance = np.linalg.norm(self.path[ind] - start_point)
            if distance >= self.lookahead_distance:
                return self.path[ind], ind
