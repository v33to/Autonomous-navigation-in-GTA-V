from gamepad import XInputDevice, AXIS_MAX, AXIS_MIN, TRIGGER_MAX
import numpy as np
import math
import time

MIN_SPEED_KMH = 5
# Minimum target speed in km/h

MAX_SPEED_KMH = 70
# Maximum target speed in km/h

MINIMUM_SPEED_TO_BRAKE = 10
# Speed threshold (km/h) below which emergency braking is disabled
# Prevents hard braking when already moving slowly

STEERING_GAIN = 0.8
# Multiplier for steering correction strength
# Lower = gentler steering, higher = more aggressive corrections

STEERING_SMOOTHING = 0.8
# How much to blend previous steering with new steering

ANGULAR_ERROR_WEIGHT = 30
# How much to prioritize angular alignment vs lateral position

MAX_STEERING_LEFT = 0.1

MAX_STEERING_RIGHT = 0.9

STEERING_LEFT_COMPRESSION_MAX = 0.35
# Reduces left steering sensitivity

STEERING_RIGHT_COMPRESSION_MIN = 0.65
# Reduces right steering sensitivity

SPEED_GAIN = 0.8
# How aggressively to slow down for trajectory errors

SPEED_SMOOTHING = 0.75
# How much to blend previous target speed with new target speed (0-1)

ERROR_NORMALIZATION_FACTOR = 30.0
# Divides total trajectory error to normalize it

THROTTLE_MIN_ACTIVATION = 0.3
# Minimum acceleration value to trigger action in game

BRAKE_MIN_ACTIVATION = 0.3
# Minimum brake value to trigger action in game

SPEED_ERROR_DEADBAND = 2.0
# Speed difference (km/h) within which no throttle/brake is applied

THROTTLE_SCALING_FACTOR = 15.0
# Divides speed error to calculate throttle amount

BRAKE_SCALING_FACTOR = 20.0
# Divides speed error to calculate brake amount

LOW_SPEED_BOOST_THRESHOLD = 20
# Speed (km/h) below which to apply minimum throttle boost

LOW_SPEED_BOOST_AMOUNT = 0.5
# Minimum throttle to apply when below LOW_SPEED_BOOST_THRESHOLD

VERY_LOW_SPEED_THRESHOLD = 1
# Speed (km/h) considered "stopped" for brake detection
# Used to detect if hard braking caused a full stop

RECOVERY_SPEED_THRESHOLD = 5
# Speed (km/h) below which recovery acceleration is applied
# After stopping, this helps get moving again

STOPPED_RECOVERY_TIME = 0.1
# Seconds to wait after stopping before applying recovery acceleration
# Prevents immediate restart after intentional stop

RECOVERY_ACCELERATION = 0.7
# Throttle amount for recovery from stopped state

SEGMENTATION_STEERING_SMOOTHING = 0.2
# Smoothing factor when segmentation requests steering change (0-1)

BASE_SEGMENTATION_CORRECTION = 0.1
# Minimum steering correction for segmentation obstacles
# Base amount to steer away from detected hazards

MAX_SEGMENTATION_CORRECTION = 0.7
# Maximum steering correction for segmentation obstacles
# Scales with obstacle proximity/severity

OBSTACLE_COOLDOWN = 0.5
# Seconds to maintain avoidance maneuver after obstacle detected
# Allows complete avoidance before going back to route following

BRAKE_HOLD_MIN_SPEED_KMH = 10.0
# Speed (km/h) below which braking is no longer forced even if an obstacle
# is still present in the ROI.  Prevents locking the vehicle at a complete
# standstill; once the car slows to this threshold the brake hold releases
# and normal navigation resumes.

BRAKE_RELEASE_GRACE_SECONDS = 0.15
# Short grace window (seconds) kept after the obstacle leaves the ROI or
# the speed threshold is met before route-following acceleration is restored.
# Absorbs 1-2 frames of detection latency so the car does not immediately
# snap back to full throttle the instant the segmentation mask clears.

AVOIDANCE_STEERING_LEFT = 0.2
# Steering value when avoiding by going left

AVOIDANCE_STEERING_RIGHT = 0.8
# Steering value when avoiding by going right

AVOIDANCE_STEERING_LEFT_LIMIT = 0.3
# Maximum steering allowed when in left avoidance mode
# Prevents over-correcting during left avoidance

AVOIDANCE_STEERING_RIGHT_LIMIT = 0.7
# Minimum steering allowed when in right avoidance mode
# Prevents over-correcting during right avoidance

EMERGENCY_BRAKE_STRENGTH = 0.7
# Brake force for critical collision warnings (0-1)

SLOW_DOWN_SPEED_THRESHOLD = 15
# Speed (km/h) below which "slow" segmentation action uses throttle instead of coasting
# Prevents excessive slowing at low speeds

SLOW_DOWN_THROTTLE = 0.4
# Throttle to apply when "slow" action is triggered at low speed
# Maintains forward progress while reducing speed

TRAJECTORY_POINTS_TO_COMPARE = 10
# Maximum number of trajectory points to compare for error calculation

MIN_TRAJECTORY_POINTS = 2
# Minimum points needed for valid trajectory error calculation

class Control:
    def __init__(self, port):
        self.gamepad = XInputDevice(port)
        self.gamepad.PlugIn()
        self.steering = 0.5
        self.acceleration = 0
        self.brake = 0
        
        self.min_speed_kmh = MIN_SPEED_KMH
        self.max_speed_kmh = MAX_SPEED_KMH
        self.steering_smoothing = STEERING_SMOOTHING
        self.speed_smoothing = SPEED_SMOOTHING
        self.steering_gain = STEERING_GAIN
        self.speed_gain = SPEED_GAIN
        self.minimum_speed_to_brake = MINIMUM_SPEED_TO_BRAKE
        
        # State tracking
        self.prev_steering = 0.5
        self.prev_target_speed = 20
        self.prev_net_throttle = 0
        self.previous_controls = {'steering': 0.5, 'acceleration': 0, 'brake': 0}
        
        self.last_steering_direction = None
        self.segmentation_steering_smoothing = SEGMENTATION_STEERING_SMOOTHING
        self.base_correction = BASE_SEGMENTATION_CORRECTION
        self.max_correction = MAX_SEGMENTATION_CORRECTION
        
        self.last_obstacle_time = 0
        self.obstacle_cooldown = OBSTACLE_COOLDOWN
        self.last_stopped_time = 0
        self.stopped_recovery_time = STOPPED_RECOVERY_TIME
        self.is_obstacle_avoidance_active = False
        self.avoidance_direction = None

        # Brake hold state
        # Suppresses route-following acceleration for as long as:
        #   (a) a braking signal is actively being received this frame, AND
        #   (b) current speed > BRAKE_HOLD_MIN_SPEED_KMH
        # A short grace window (BRAKE_RELEASE_GRACE_SECONDS) is kept after
        # both conditions clear to absorb segmentation latency.
        self.last_brake_trigger_time = 0.0
        self.brake_hold_active = False
        self.brake_hold_min_speed = BRAKE_HOLD_MIN_SPEED_KMH
        self.brake_release_grace = BRAKE_RELEASE_GRACE_SECONDS

    def reset(self):
        """Reset all controls to neutral and clear state"""
        self.gamepad.SetAxis('X', 0)
        self.gamepad.SetTrigger('R', 0)
        self.gamepad.SetTrigger('L', 0)
        self.steering = 0.5
        self.acceleration = 0
        self.brake = 0
        self.last_obstacle_time = 0
        self.last_stopped_time = 0
        self.is_obstacle_avoidance_active = False
        self.avoidance_direction = None
        self.prev_net_throttle = 0
        self.last_brake_trigger_time = 0.0
        self.brake_hold_active = False

    def calculate_trajectory_error(self, current_trajectory, desired_trajectory):
        """
        Calculate lateral and angular errors between trajectories
        Returns: (lateral_error, angular_error, total_error)
        """
        if not current_trajectory or not desired_trajectory:
            return 0, 0, 0
        
        min_points = min(len(current_trajectory), len(desired_trajectory), TRAJECTORY_POINTS_TO_COMPARE)
        if min_points < MIN_TRAJECTORY_POINTS:
            return 0, 0, 0
        
        lateral_errors = []
        angular_errors = []
        
        for i in range(min_points):
            if i >= len(desired_trajectory):
                break
                
            curr_point = current_trajectory[i]
            desired_point = desired_trajectory[i]
            
            # Lateral error (x difference)
            lateral_error = curr_point[0] - desired_point[0]
            lateral_errors.append(lateral_error)
            
            # Angular error if we have next points
            if i < min_points - 1 and i < len(desired_trajectory) - 1:
                curr_next = current_trajectory[i + 1] if i + 1 < len(current_trajectory) else current_trajectory[i]
                curr_dir = math.atan2(curr_next[1] - curr_point[1], curr_next[0] - curr_point[0])
                
                desired_next = desired_trajectory[i + 1]
                desired_dir = math.atan2(desired_next[1] - desired_point[1], desired_next[0] - desired_point[0])
                
                angular_diff = curr_dir - desired_dir
                angular_diff = math.atan2(math.sin(angular_diff), math.cos(angular_diff))
                angular_errors.append(angular_diff)
        
        avg_lateral = np.mean(lateral_errors) if lateral_errors else 0
        avg_angular = np.mean(angular_errors) if angular_errors else 0
        total_error = abs(avg_lateral) + abs(avg_angular) * 20
        
        return avg_lateral, avg_angular, total_error

    def calculate_navigation_controls(self, current_trajectory, desired_trajectory, current_speed):
        """
        Calculate steering, acceleration and brake based on trajectory error
        Returns: (steering, acceleration, brake)
        """
        lateral_error, angular_error, total_error = self.calculate_trajectory_error(
            current_trajectory, desired_trajectory)
        
        # Steering calculation
        combined_error = lateral_error + angular_error * ANGULAR_ERROR_WEIGHT
        steering_adjustment = -combined_error * self.steering_gain / 90.0
        steering_adjustment = max(-0.5, min(0.5, steering_adjustment))
        target_steering = 0.5 + steering_adjustment
        
        # Apply steering smoothing
        smoothed_steering = (self.steering_smoothing * self.prev_steering + 
                           (1 - self.steering_smoothing) * target_steering)
        
        if smoothed_steering > 0.5:
            # Compress right steering range
            if smoothed_steering > STEERING_RIGHT_COMPRESSION_MIN:
                smoothed_steering = STEERING_RIGHT_COMPRESSION_MIN + (
                    (smoothed_steering - STEERING_RIGHT_COMPRESSION_MIN) / 
                    (1.0 - STEERING_RIGHT_COMPRESSION_MIN)
                ) * (MAX_STEERING_RIGHT - STEERING_RIGHT_COMPRESSION_MIN)
        else:
            # Compress left steering range
            if smoothed_steering < STEERING_LEFT_COMPRESSION_MAX:
                smoothed_steering = STEERING_LEFT_COMPRESSION_MAX - (
                    (STEERING_LEFT_COMPRESSION_MAX - smoothed_steering) / 
                    STEERING_LEFT_COMPRESSION_MAX
                ) * (STEERING_LEFT_COMPRESSION_MAX - MAX_STEERING_LEFT)
        
        smoothed_steering = max(MAX_STEERING_LEFT, min(MAX_STEERING_RIGHT, smoothed_steering))
        self.prev_steering = smoothed_steering
        
        # Calculate target speed based on trajectory error
        normalized_error = min(total_error / ERROR_NORMALIZATION_FACTOR, 1.0)
        speed_range = self.max_speed_kmh - self.min_speed_kmh
        target_speed = self.max_speed_kmh - (normalized_error * speed_range * self.speed_gain)
        target_speed = max(self.min_speed_kmh, min(self.max_speed_kmh, target_speed))
        
        smoothed_target_speed = (self.speed_smoothing * self.prev_target_speed + 
                               (1 - self.speed_smoothing) * target_speed)
        self.prev_target_speed = smoothed_target_speed
        speed_error = smoothed_target_speed - current_speed
        
        if abs(speed_error) < SPEED_ERROR_DEADBAND:
            # Within deadband - coast
            acceleration = 0.0
            brake = 0.0
        elif speed_error > 0:  # Need to accelerate
            acceleration = min(speed_error / THROTTLE_SCALING_FACTOR, 1.0)
            acceleration = max(THROTTLE_MIN_ACTIVATION, acceleration)
            brake = 0.0
        else:  # Need to brake
            brake = min(abs(speed_error) / BRAKE_SCALING_FACTOR, 1.0)
            brake = max(BRAKE_MIN_ACTIVATION, brake)
            acceleration = 0.0
        
        return smoothed_steering, acceleration, brake

    def apply_navigation_controls(self, current_trajectory, desired_trajectory, current_speed, 
                                 collision_warnings=None, segmentation_action=None):
        """
        Apply navigation-based controls with segmentation override capability.
        Main control loop that integrates trajectory following with safety overrides.
        """
        current_time = time.time()
        
        # Store previous controls for segmentation smoothing
        self.previous_controls = {
            'steering': self.steering,
            'acceleration': self.acceleration,
            'brake': self.brake
        }
        
        steering, acceleration, brake = self.calculate_navigation_controls(
            current_trajectory, desired_trajectory, current_speed)
        
        self.steering = steering
        self.acceleration = acceleration
        self.brake = brake

        # Check segmentation action
        seg_brake_this_frame = (
            segmentation_action is not None and
            segmentation_action.get("speed") == "stop"
        )
        # Check collision warnings (CRITICAL warnings trigger emergency brake)
        critical_brake_this_frame = bool(
            collision_warnings and
            any(("CRITICAL" in w and ("Person" in w or "Car" in w))
                for w in collision_warnings)
        )
        brake_signal_this_frame = seg_brake_this_frame or critical_brake_this_frame

        # Refresh the "last seen" timestamp while the signal is present
        if brake_signal_this_frame:
            self.last_brake_trigger_time = current_time

        within_grace = (current_time - self.last_brake_trigger_time) < self.brake_release_grace
        above_hold_threshold = current_speed > self.brake_hold_min_speed
        brake_hold_active = (brake_signal_this_frame or within_grace) and above_hold_threshold

        if brake_hold_active:
            self.acceleration = 0.0
        
        # Boost throttle at low speeds to prevent crawling.
        # Skipped while brake hold is active to avoid fighting a braking command.
        if not brake_hold_active and current_speed < LOW_SPEED_BOOST_THRESHOLD and self.acceleration >= 0:
            self.acceleration = max(self.acceleration, LOW_SPEED_BOOST_AMOUNT)
        
        # Detect if we've stopped due to hard braking
        if current_speed < VERY_LOW_SPEED_THRESHOLD and self.brake > 0.5:
            self.last_stopped_time = current_time
        
        # Recovery acceleration after stopping.
        # Also gated by brake hold so we do not instantly re-accelerate into an obstacle.
        if (not brake_hold_active and
                current_speed < RECOVERY_SPEED_THRESHOLD and
                (current_time - self.last_stopped_time) > self.stopped_recovery_time):
            # Check if there's a critical obstacle ahead
            critical_ahead = False
            if collision_warnings:
                for warning in collision_warnings:
                    if "CRITICAL" in warning and "lower" in warning:
                        critical_ahead = True
                        break
            
            # If no critical obstacle, apply recovery acceleration
            if not critical_ahead:
                self.acceleration = RECOVERY_ACCELERATION
                self.brake = 0
        
        # Deactivate obstacle avoidance mode after cooldown period
        if self.is_obstacle_avoidance_active and (current_time - self.last_obstacle_time) > self.obstacle_cooldown:
            self.is_obstacle_avoidance_active = False
            self.avoidance_direction = None
        
        # Apply steering and speed adjustments based on semantic segmentation
        if segmentation_action and not self.is_obstacle_avoidance_active:
            offset = segmentation_action.get("offset", 0)
            # Scale correction based on obstacle proximity
            adaptive_correction = self.base_correction + (self.max_correction - self.base_correction) * min(abs(offset), 1.0)
            
            # Steering overrides
            if segmentation_action["steer"] == "left":
                target_steering = max(0.1, self.steering - adaptive_correction)
                self.steering = (self.previous_controls['steering'] * (1 - self.segmentation_steering_smoothing) + 
                               target_steering * self.segmentation_steering_smoothing)
                self.last_steering_direction = "left"
                self.is_obstacle_avoidance_active = True
                self.avoidance_direction = "left"
                self.last_obstacle_time = current_time
                
            elif segmentation_action["steer"] == "right":
                target_steering = min(0.9, self.steering + adaptive_correction)
                self.steering = (self.previous_controls['steering'] * (1 - self.segmentation_steering_smoothing) + 
                               target_steering * self.segmentation_steering_smoothing)
                self.last_steering_direction = "right"
                self.is_obstacle_avoidance_active = True
                self.avoidance_direction = "right"
                self.last_obstacle_time = current_time
            
            # Speed overrides
            if segmentation_action["speed"] == "stop":
                if current_speed > self.minimum_speed_to_brake:
                    self.acceleration = 0
                    self.brake = EMERGENCY_BRAKE_STRENGTH
                elif current_speed < VERY_LOW_SPEED_THRESHOLD:
                    self.last_stopped_time = current_time
                    
            elif segmentation_action["speed"] == "slow":
                if current_speed > SLOW_DOWN_SPEED_THRESHOLD:
                    self.acceleration = 0  # Coast to slow down
                else:
                    # At low speed, maintain minimum throttle
                    self.acceleration = max(SLOW_DOWN_THROTTLE, self.acceleration)
                    self.brake = 0
        
        # Emergency braking and avoidance for detected collisions
        if collision_warnings:
            critical_pedestrian_warnings = [w for w in collision_warnings if "CRITICAL" in w and "Person" in w]
            critical_vehicle_warnings = [w for w in collision_warnings if "CRITICAL" in w and "Car" in w]
            
            # Emergency stop for critical warnings
            if (critical_pedestrian_warnings or critical_vehicle_warnings) and current_speed > self.minimum_speed_to_brake:
                self.acceleration = 0
                self.brake = EMERGENCY_BRAKE_STRENGTH
            
            # Avoidance steering if not already in avoidance mode
            if not self.is_obstacle_avoidance_active:
                lower_ped_warnings = [w for w in critical_pedestrian_warnings if "lower" in w]
                lower_car_warnings = [w for w in critical_vehicle_warnings if "lower" in w]
                
                # Check pedestrian warnings
                if lower_ped_warnings:
                    for warning in lower_ped_warnings:
                        if "left" in warning:
                            self.steering = AVOIDANCE_STEERING_RIGHT
                            self.is_obstacle_avoidance_active = True
                            self.avoidance_direction = "right"
                            self.last_obstacle_time = current_time
                        elif "right" in warning:
                            self.steering = AVOIDANCE_STEERING_LEFT
                            self.is_obstacle_avoidance_active = True
                            self.avoidance_direction = "left"
                            self.last_obstacle_time = current_time
                
                # Check vehicle warnings
                elif lower_car_warnings:
                    for warning in lower_car_warnings:
                        if "left" in warning:
                            self.steering = AVOIDANCE_STEERING_RIGHT
                            self.is_obstacle_avoidance_active = True
                            self.avoidance_direction = "right"
                            self.last_obstacle_time = current_time
                        elif "right" in warning:
                            self.steering = AVOIDANCE_STEERING_LEFT
                            self.is_obstacle_avoidance_active = True
                            self.avoidance_direction = "left"
                            self.last_obstacle_time = current_time
        
        # Constrain steering during active avoidance maneuvers
        if self.is_obstacle_avoidance_active:
            if self.avoidance_direction == "left":
                self.steering = min(self.steering, AVOIDANCE_STEERING_LEFT_LIMIT)
            elif self.avoidance_direction == "right":
                self.steering = max(self.steering, AVOIDANCE_STEERING_RIGHT_LIMIT)
        
        # Convert normalized values to Xbox controller ranges
        xbox_steering = int(self.steering * (AXIS_MAX - AXIS_MIN) + AXIS_MIN)
        self.gamepad.SetAxis('X', xbox_steering)
        
        xbox_acceleration = int(self.acceleration * TRIGGER_MAX)
        self.gamepad.SetTrigger('R', xbox_acceleration)
        
        xbox_braking = int(self.brake * TRIGGER_MAX)
        self.gamepad.SetTrigger('L', xbox_braking)