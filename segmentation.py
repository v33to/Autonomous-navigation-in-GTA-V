import torch
import numpy as np
import cv2
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Real-time semantic segmentation system for assistance. Processes road images to detect vehicles, pedestrians 
and road boundaries, providing collision warnings and steering recommendations based on detected hazards.
"""

path = os.getcwd()

BOUNDARY_MARGIN: int   = 40
BOUNDARY_INTERVENTION_THRESHOLD: float = 0.6

_PED_CLASS_ID    = 9
_PED_OTHER_FG    = {1, 10}          # foreground classes that must not be overwritten
_DILATION_RULES  = [(30, 13), (250, 9)]
_DILATION_KERNELS: dict = {
    ksize: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    for _, ksize in _DILATION_RULES
}


def _dilate_pedestrians(prediction: np.ndarray) -> np.ndarray:
    """
    Return a copy of *prediction* with per-component adaptive dilation applied
    to the pedestrian class (ID 9). Components with area >= 250 px are left unchanged.
    """
    ped_mask = (prediction == _PED_CLASS_ID).astype(np.uint8)
    if not ped_mask.any():
        return prediction

    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        ped_mask, connectivity=8
    )

    other_fg = np.zeros(prediction.shape, dtype=bool)
    for cid in _PED_OTHER_FG:
        other_fg |= (prediction == cid)

    dilated_ped = np.zeros_like(ped_mask)
    for label_id in range(1, num_labels):
        area  = int(stats[label_id, cv2.CC_STAT_AREA])
        ksize = next((k for thr, k in _DILATION_RULES if area < thr), None)
        if ksize is None:
            continue
        component   = (label_map == label_id).astype(np.uint8)
        dilated_ped = np.maximum(dilated_ped,
                                 cv2.dilate(component, _DILATION_KERNELS[ksize]))

    dilated_ped[other_fg] = 0
    out = prediction.copy()
    out[dilated_ped > 0] = _PED_CLASS_ID
    return out


class Segmentation:
    def __init__(self):
        """
        Initialize the segmentation system with model parameters, class definitions
        and region of interest settings for collision detection.
        """
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = 0.5
        self.ema_probs = None

        self.class_colors = {
            0: (0, 0, 0),        # unlabeled
            1: (128, 64, 128),   # road
            2: (244, 35, 232),   # sidewalk
            3: (70, 70, 70),     # building
            4: (153, 153, 153),  # pole
            5: (250, 170, 30),   # traffic light
            6: (107, 142, 35),   # vegetation
            7: (152, 251, 152),  # terrain
            8: (70, 130, 180),   # sky
            9: (220, 20, 60),    # person
            10: (0, 0, 142),     # car
        }
        
        self.class_priorities = {
            9: 3,   # person
            10: 2,  # car
            1: 0,   # road
            2: 1,   # sidewalk
            3: 1,   # building
            4: 1,   # pole
            5: 1,   # traffic light
            6: 1,   # vegetation
            7: 1,   # terrain
            0: 0,   # unlabeled
            8: 0,   # sky
        }
        self.trapezoid_roi = None
        self.mean = [0.383, 0.376, 0.358]
        self.std = [0.221, 0.209, 0.190]
        self.roi = (0, 117, 1152, 693)
        self.final_size = (512, 256)
        self.segmentation_active = False
        self.visualization_active = False
        self.model_loaded = False
        self.model = None
        self.window_name = 'Semantic Segmentation'
        self.window_width = 750
        self.window_x = 1154
        self.window_y = 240
        
        self.collision_warnings = []
        self.action_taken = None
        self.danger_zones = {}
        self.road_boundaries = None
        self.original_image_shape = None

        self.base_trapezoid_roi = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [0.65, 0.8],
            [0.35, 0.8]
        ])

        self.max_trapezoid_roi = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [0.65, 0.6],
            [0.35, 0.6]
        ])

        self.current_trapezoid_roi = self.base_trapezoid_roi.copy()
        self.roi_speed_factor = 0.01
        self.max_speed_roi = 70
        self.roi_upper_threshold = 0.2
        self.base_trapezoid_roi_scaled = None

    def load_segmentation_model(self):
        """
        Load the segmentation model and prepare it for inference.
        """
        try:
            model_path = self.path + "/Segmentation/segmentation_ddrnet.pt"
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_loaded = True
            print("Segmentation model loaded successfully.")
        except Exception as e:
            print(f"Error loading segmentation model. Check path and file.")
            print(f"Error details: {str(e)}")
            exit()

    def warm_up_model(self):
        """
        Warm up the model by running a dummy inference to optimize performance.
        """
        if not self.model_loaded:
            self.load_segmentation_model()

        try:
            dummy_input = torch.zeros(1, 3, self.final_size[1], self.final_size[0], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        except Exception as e:
            print(f"Error warming up segmentation model: {str(e)}")

    def get_prediction(self, image):
        """
        Run segmentation inference on a raw captured frame and return the raw class-index map.
        """
        if not self.model_loaded:
            self.load_segmentation_model()

        x1, y1, x2, y2 = self.roi
        image_roi = image[y1:y2, x1:x2]
        self.original_image_shape = image_roi.shape[:2]

        transform = A.Compose([
            A.Resize(height=self.final_size[1], width=self.final_size[0]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        transformed = transform(image=image_roi)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)

        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        prediction = self.ema_probs.argmax(dim=1).cpu().numpy()[0]
        return _dilate_pedestrians(prediction)

    def process_image_from_prediction(self, image, prediction, speed=0):
        """
        Perform collision analysis and optional visualization using a prediction
        map that was already computed by get_prediction().
        """
        if not self.segmentation_active:
            return None, [], None

        if self.visualization_active:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.visualization_active = False
            except:
                self.visualization_active = False

        self.update_roi_for_speed(speed)

        self.collision_warnings, action, self.danger_zones, self.road_boundaries = \
            self.analyze_prediction(prediction)

        if self.visualization_active:
            x1, y1, x2, y2 = self.roi
            self.visualize_segmentation(image[y1:y2, x1:x2], prediction)

        return prediction, self.collision_warnings, action

    def toggle_segmentation(self, enable_visualization=True):
        """
        Toggle the segmentation system on/off, initializing the model and visualization window
        when activated or closing them when deactivated.
        """
        if not self.segmentation_active:
            if not self.model_loaded:
                self.load_segmentation_model()
            self.warm_up_model()
            if enable_visualization:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.window_width, 304)
                cv2.moveWindow(self.window_name, self.window_x, self.window_y)
                self.visualization_active = True
            else:
                self.visualization_active = False
                print("Segmentation processing enabled (no visualization)")
        else:
            if self.visualization_active:
                self.close_segmentation_window()
            self.ema_probs = None

        self.segmentation_active = not self.segmentation_active

    def close_segmentation_window(self):
        """
        Close the segmentation visualization window.
        """
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(self.window_name)
        except:
            pass
        self.visualization_active = False

    def update_roi_for_speed(self, speed):
        """
        Dynamically adjust the region of interest based on vehicle speed.
        Higher speeds expand the ROI to detect objects further ahead.
        """
        if speed <= 0:
            self.current_trapezoid_roi = self.base_trapezoid_roi.copy()
            return
            
        factor = min(speed / self.max_speed_roi, 1.0)
        self.current_trapezoid_roi = self.base_trapezoid_roi.copy()
        for i in range(2, 4):
            self.current_trapezoid_roi[i] = (self.base_trapezoid_roi[i] * (1 - factor) + self.max_trapezoid_roi[i] * factor)

    def process_image(self, image, speed=0):
        """
        Main processing function that performs segmentation inference on the input image,
        analyzes the results for collision risks and returns predictions and warnings.
        """
        if not self.segmentation_active:
            return None, [], None
        
        if self.visualization_active:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.visualization_active = False
            except:
                self.visualization_active = False

        self.update_roi_for_speed(speed)
        x1, y1, x2, y2 = self.roi
        image_roi = image[y1:y2, x1:x2]
        self.original_image_shape = image_roi.shape[:2]

        transform = A.Compose([
            A.Resize(height=self.final_size[1], width=self.final_size[0]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        transformed = transform(image=image_roi)
        image_tensor = transformed['image']
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)

        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs
        
        prediction = self.ema_probs.argmax(dim=1).cpu().numpy()[0]
        prediction = _dilate_pedestrians(prediction)
        self.collision_warnings, action, self.danger_zones, self.road_boundaries = self.analyze_prediction(prediction)
        if self.visualization_active:
            self.visualize_segmentation(image_roi, prediction)
        
        return prediction, self.collision_warnings, action

    def analyze_prediction(self, prediction):
        """
        Analyze the segmentation prediction to identify collision risks and recommend actions.
        Detects persons and cars within both the base ROI (steering + braking) and the
        expanded ROI (braking only), tagging each detection with its source zone so that
        determine_action() can apply the correct set of signals.
        """
        h, w = prediction.shape
        warnings = []
        danger_zones = {}
        road_boundaries = self.detect_road_boundaries(prediction)

        # Build base ROI
        base_roi_scaled = self.base_trapezoid_roi.copy()
        base_roi_scaled[:, 0] *= w
        base_roi_scaled[:, 1] *= h
        base_roi_scaled = base_roi_scaled.astype(np.int32)
        self.base_trapezoid_roi_scaled = base_roi_scaled

        base_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(base_roi_mask, [base_roi_scaled], 1)

        # Build expanded ROI
        expanded_roi_scaled = self.current_trapezoid_roi.copy()
        expanded_roi_scaled[:, 0] *= w
        expanded_roi_scaled[:, 1] *= h
        expanded_roi_scaled = expanded_roi_scaled.astype(np.int32)
        self.trapezoid_roi = expanded_roi_scaled

        expanded_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(expanded_roi_mask, [expanded_roi_scaled], 1)

        # Region that belongs exclusively to the expanded ROI
        expanded_only_mask = cv2.bitwise_and(
            expanded_roi_mask,
            cv2.bitwise_not(base_roi_mask)
        )

        # Combined mask for initial presence check
        combined_roi_mask = cv2.bitwise_or(base_roi_mask, expanded_roi_mask)

        # Upper/lower sub-regions apply only within the base ROI
        upper_threshold = int(h * self.roi_upper_threshold)
        upper_base_mask = base_roi_mask.copy()
        upper_base_mask[upper_threshold:, :] = 0
        lower_base_mask = base_roi_mask.copy()
        lower_base_mask[:upper_threshold, :] = 0

        for obj_class in [9, 10]:
            obj_mask = (prediction == obj_class).astype(np.uint8)
            intersection = cv2.bitwise_and(obj_mask, combined_roi_mask)

            if np.sum(intersection) > 50:
                class_name = "Person" if obj_class == 9 else "Car"
                num_labels, labels = cv2.connectedComponents(obj_mask)
                danger_zones[obj_class] = []

                for label in range(1, num_labels):
                    obj_indices = np.where(labels == label)

                    if len(obj_indices[0]) > 50:
                        component_mask = (labels == label).astype(np.uint8)
                        comp_intersection = cv2.bitwise_and(component_mask, combined_roi_mask)

                        if np.sum(comp_intersection) > 25:
                            top = np.min(obj_indices[0])
                            bottom = np.max(obj_indices[0])
                            left = np.min(obj_indices[1])
                            right = np.max(obj_indices[1])
                            distance = 1.0 - (np.max(obj_indices[0]) / h)

                            center_x = (left + right) // 2
                            image_center = w // 2
                            position = "left" if center_x < image_center else "right"

                            # upper/lower sub-regions (base ROI only)
                            in_upper_half = np.sum(cv2.bitwise_and(component_mask, upper_base_mask)) > 0
                            in_lower_half = np.sum(cv2.bitwise_and(component_mask, lower_base_mask)) > 0

                            region = []
                            if in_upper_half:
                                region.append("upper")
                            if in_lower_half:
                                region.append("lower")
                            region_str = "+".join(region)

                            # Determine which ROI zone the detection belongs to
                            in_base     = np.sum(cv2.bitwise_and(component_mask, base_roi_mask)) > 0
                            in_expanded = np.sum(cv2.bitwise_and(component_mask, expanded_only_mask)) > 0

                            if in_base:
                                roi_zone = "base"
                            else:
                                roi_zone = "expanded"
                                
                            danger_zones[obj_class].append(
                                (top, left, bottom, right, position, distance, region_str, roi_zone)
                            )

                            if distance < 0.4:
                                warnings.append(f"CRITICAL: {class_name} detected in {position} region!")
                            elif distance < 0.6:
                                warnings.append(f"WARNING: {class_name} approaching in {position} region")
        
        object_positions = self.get_object_positions(danger_zones)
        action = self.determine_action(object_positions, road_boundaries)
        
        return warnings, action, danger_zones, road_boundaries

    def get_object_positions(self, danger_zones):
        """
        Extract object positions for action determination, prioritizing closest objects.
        Includes the roi_zone tag ('base' or 'expanded') so that determine_action() can
        restrict expanded-zone detections to braking signals only.
        """
        object_positions = {}

        for obj_class, objects in danger_zones.items():
            if objects:
                sorted_objects = sorted(objects, key=lambda x: x[5])
                closest_obj = sorted_objects[0]
                object_positions[obj_class] = {
                    'position': closest_obj[4],
                    'region':   closest_obj[6],
                    'distance': closest_obj[5],
                    'roi_zone': closest_obj[7] if len(closest_obj) > 7 else 'base',
                }

        return object_positions

    def determine_action(self, object_positions, road_boundaries):
        """
        Determine what action to take based on detected objects and road boundaries.
        Priority: pedestrian safety > vehicle avoidance > road boundaries.

        ROI zone rules
        --------------
        base     – obstacle is inside the base (near) trapezoid  → steering + braking allowed
        expanded – obstacle is only in the expanded (far) region  → braking only, no steering
        """
        action = {
            "steer": "maintain",
            "speed": "maintain",
            "priority": 0,
            "offset": 0
        }

        if 9 in object_positions:
            ped_info   = object_positions[9]
            ped_pos    = ped_info['position']
            ped_region = ped_info['region']
            ped_zone   = ped_info.get('roi_zone', 'base')

            if ped_zone == 'expanded':
                new_action = {
                    "steer":    "maintain",
                    "speed":    "slow",
                    "priority": 3
                }
            elif 'lower' in ped_region:
                new_action = {
                    "steer":    "right" if ped_pos == "left" else "left" if ped_pos == "right" else "maintain",
                    "speed":    "stop",
                    "priority": 3
                }
            else:
                new_action = {
                    "steer":    "maintain",
                    "speed":    "stop",
                    "priority": 3
                }

            if new_action["priority"] > action["priority"]:
                action = new_action

        if 10 in object_positions and action["priority"] < 3:
            car_info     = object_positions[10]
            car_pos      = car_info['position']
            car_region   = car_info['region']
            car_distance = car_info['distance']
            car_zone     = car_info.get('roi_zone', 'base')

            if car_zone == 'expanded':
                new_action = {
                    "steer":    "maintain",
                    "speed":    "slow",
                    "priority": 2
                }
            elif 'lower' in car_region:
                if car_distance < 0.3:
                    new_action = {
                        "steer":    "right" if car_pos == "left" else "left" if car_pos == "right" else "maintain",
                        "speed":    "stop",
                        "priority": 2
                    }
                else:
                    new_action = {
                        "steer":    "right" if car_pos == "left" else "left" if car_pos == "right" else "maintain",
                        "speed":    "slow",
                        "priority": 2,
                        "offset":   car_info.get('offset', 0)
                    }
            else:
                new_action = {
                    "steer":    "maintain",
                    "speed":    "slow",
                    "priority": 1
                }

            if new_action["priority"] > action["priority"]:
                action = new_action
        
        if road_boundaries and action["priority"] < 2:
            left_boundary, right_boundary, center_offset = road_boundaries
            action["offset"] = center_offset

            if left_boundary is not None and right_boundary is not None:
                w          = self.final_size[0]
                vehicle_x  = w / 2

                # Absolute boundary gate; the "safe zone" is the road interior shrunk by BOUNDARY_MARGIN on each side.
                safe_left  = left_boundary  + BOUNDARY_MARGIN
                safe_right = right_boundary - BOUNDARY_MARGIN

                half_safe  = max((safe_right - safe_left) / 2.0, 1.0)
                road_center_x = (safe_left + safe_right) / 2.0

                # Normalised proximity: 0 at road centre → 1 at the margin line → >1 outside.
                proximity = abs(vehicle_x - road_center_x) / half_safe

                if proximity >= BOUNDARY_INTERVENTION_THRESHOLD:
                    # Direction: steer away from whichever boundary is closer
                    steer_dir = "right" if vehicle_x < road_center_x else "left"

                    new_action = {
                        "steer":    steer_dir,
                        "speed":    "maintain",
                        "priority": 1,
                        "offset":   min(proximity, 1.0),
                    }
                    if new_action["priority"] > action["priority"]:
                        action = new_action
        
        self.action_taken = action
        return action

    def detect_road_boundaries(self, prediction):
        """
        Detect road boundaries from segmentation and calculate offset from center
        """
        h, w = prediction.shape
        road_mask = prediction == 1
        
        if not np.any(road_mask):
            return None, None, 0
        
        search_region = road_mask[2*h//3:, :]
        left_boundaries = []
        right_boundaries = []
        
        for row in search_region:
            road_indices = np.where(row)[0]
            if len(road_indices) > 0:
                left_boundaries.append(road_indices[0])
                right_boundaries.append(road_indices[-1])
        
        left_boundary = int(np.median(left_boundaries)) if left_boundaries else None
        right_boundary = int(np.median(right_boundaries)) if right_boundaries else None
        
        center_offset = 0
        if left_boundary is not None and right_boundary is not None:
            road_center = (left_boundary + right_boundary) // 2
            image_center = w // 2
            center_offset = (road_center - image_center) / (w / 2)
        
        return left_boundary, right_boundary, center_offset

    def visualize_segmentation(self, image, segmentation):
        """
        Create and display a visualization overlay showing segmentation results,
        detected objects, region of interest and recommended actions.
        """
        if not self.visualization_active:
            return
        segmentation_vis = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            mask = segmentation == int(class_idx)
            segmentation_vis[mask] = color
        
        segmentation_vis = cv2.resize(segmentation_vis, (image.shape[1], image.shape[0]))
        segmentation_vis = cv2.cvtColor(segmentation_vis, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = 0.6
        overlay_image = cv2.addWeighted(image, 1 - alpha, segmentation_vis, alpha, 0)
        
        if self.trapezoid_roi is not None:
            scale_y = image.shape[0] / self.final_size[1]
            scale_x = image.shape[1] / self.final_size[0]

            # Draw expanded ROI (braking only) in yellow
            scaled_expanded = self.trapezoid_roi.copy().astype(np.float32)
            scaled_expanded[:, 0] *= scale_x
            scaled_expanded[:, 1] *= scale_y
            scaled_expanded = scaled_expanded.astype(np.int32)
            cv2.polylines(overlay_image, [scaled_expanded], True, (255, 255, 0), 2)
            cv2.putText(overlay_image, "Expanded ROI (brake only)",
                        (scaled_expanded[3, 0], scaled_expanded[3, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw base ROI (steering + braking) in cyan
            if hasattr(self, 'base_trapezoid_roi_scaled'):
                scaled_base = self.base_trapezoid_roi_scaled.copy().astype(np.float32)
                scaled_base[:, 0] *= scale_x
                scaled_base[:, 1] *= scale_y
                scaled_base = scaled_base.astype(np.int32)
                cv2.polylines(overlay_image, [scaled_base], True, (0, 255, 255), 2)
                cv2.putText(overlay_image, "Base ROI (steer + brake)",
                            (scaled_base[3, 0], scaled_base[3, 1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        if self.road_boundaries and self.road_boundaries[0] is not None and self.road_boundaries[1] is not None:
            scale_x = image.shape[1] / self.final_size[0]
            left_boundary = int(self.road_boundaries[0] * scale_x)
            right_boundary = int(self.road_boundaries[1] * scale_x)
            
            cv2.line(overlay_image, (left_boundary, 0), (left_boundary, image.shape[0]), (0, 255, 255), 2)
            cv2.line(overlay_image, (right_boundary, 0), (right_boundary, image.shape[0]), (0, 255, 255), 2)
        
        if self.action_taken and self.action_taken["steer"] != "maintain":
            if self.action_taken["steer"] == "left":
                cv2.arrowedLine(overlay_image, 
                            (image.shape[1] // 2, image.shape[0] - 50), 
                            (image.shape[1] // 2 - 50, image.shape[0] - 50), 
                            (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(overlay_image, "TURN LEFT", 
                        (image.shape[1] // 2 - 100, image.shape[0] - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif self.action_taken["steer"] == "right":
                cv2.arrowedLine(overlay_image, 
                            (image.shape[1] // 2, image.shape[0] - 50), 
                            (image.shape[1] // 2 + 50, image.shape[0] - 50), 
                            (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(overlay_image, "TURN RIGHT", 
                        (image.shape[1] // 2 - 50, image.shape[0] - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for obj_class, objects in self.danger_zones.items():
            if objects:
                for obj_bounds in objects:
                    top, left, bottom, right, position, distance, region = obj_bounds[:7]
                    roi_zone = obj_bounds[7] if len(obj_bounds) > 7 else 'base'
                    scale_y = image.shape[0] / self.final_size[1]
                    scale_x = image.shape[1] / self.final_size[0]
                    top    = int(top    * scale_y)
                    bottom = int(bottom * scale_y)
                    left   = int(left   * scale_x)
                    right  = int(right  * scale_x)

                    class_name    = "Person" if obj_class == 9 else "Car"
                    priority_text = "HIGH PRIORITY" if self.class_priorities[obj_class] > 2 else "MEDIUM PRIORITY"
                    zone_label    = "base" if roi_zone == 'base' else "expanded"

                    if roi_zone == 'expanded':
                        color     = (0, 165, 255)
                        thickness = 2
                    else:
                        color     = (255, 0, 0) if obj_class == 9 else (0, 0, 255)
                        thickness = 3 if self.class_priorities[obj_class] > 2 else 2

                    cv2.rectangle(overlay_image, (left, top), (right, bottom), color, thickness)
                    cv2.putText(overlay_image,
                                f"{class_name} ({priority_text}) [{zone_label}]",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        y_offset = 30
        for warning in self.collision_warnings:
            color = (255, 0, 0) if "CRITICAL" in warning else (255, 0, 0) if "WARNING" in warning else (255, 0, 0)
            cv2.putText(overlay_image, warning, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        if self.action_taken:
            action_text = f"ACTION: Steer {self.action_taken['steer']}, Speed {self.action_taken['speed']}"
            cv2.putText(overlay_image, action_text, 
                        (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        try:
            cv2.imshow(self.window_name, overlay_image)
            cv2.waitKey(1)
        except:
            self.visualization_active = False