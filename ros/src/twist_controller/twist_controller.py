#!/usr/bin/env python

import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle): 
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0
        mx = 1
        
        self.throttle_controller = PID( kp, ki, kd, mn, mx)
        
        tau = 0.5
        ts = 0.02
        self.velocity_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()
        self.last_velocity = 0
        

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0, 0, 0
        
        current_velocity = self.velocity_lpf.filt(current_velocity)
        
        
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        # Get throttle from PID controller
        throttle = self.throttle_controller.step(velocity_error, sample_time)
       
        brake = 0
      
        #checking if we need to brake
        if linear_velocity == 0. and current_velocity < 0.1: #case of stopped car
            throttle = 0 # override value fro throttle controller
            brake = 400
            
        elif throttle <0.1 and velocity_error < 0:
            decel = max(velocity_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius #torque
            
        return throttle, brake, steering #throttle, brake, steering
        