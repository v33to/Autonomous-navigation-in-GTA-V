import numpy as np
import cv2
from SpeedAcquisition.test import get_speed, get_speed_from_file
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw

class ImageProcessing:
    def __init__(self, show_visualization: bool = True):
        # ROI coordinates for minimap (x1, y1, x2, y2)
        self.roi_coords = (19, 699, 234, 795)
        
        # HSV thresholds for route detection (magenta)
        self.low_thresh = np.array([130, 150, 230])
        self.high_thresh = np.array([140, 180, 255])
        
        # Trajectory extraction parameters
        self.min_segment_length = 3
        self.mask_dilation_iterations = 2  # Helps maintain route when slightly off-path
        self.mask_dilation_kernel_size = 2
        
        self.show_visualization = show_visualization
        self.window_name = "Minimap Trajectory Analysis"
        self.window_initialized = False
        self.route_detected = True

    def initialize_window(self):
        """Initialize the display window."""
        if not self.window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.window_name, 1154, 0)
            self.window_initialized = True

    def process_image(self, image):
        """
        Processes a single frame by extracting speed and minimap trajectory data.
        Returns speed, trajectory visualization, current_trajectory, desired_trajectory, route_detected.
        """
        # Extract speed
        # speed = np.float32(get_speed(image))
        speed = np.float32(get_speed_from_file())
        
        # Extract minimap ROI
        x1, y1, x2, y2 = self.roi_coords
        height, width = image.shape[:2]
        
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x2 <= x1 or y2 <= y1:
            self.route_detected = False
            return speed, None, [], [], False
            
        # Extract minimap region
        minimap = image[y1:y2, x1:x2]
        trajectory_vis, current_traj, desired_traj = self.analyze_trajectories(minimap)
        
        # Check if route was detected
        self.route_detected = len(desired_traj) > 0
        
        # Display the frame only when visualization is enabled
        if self.show_visualization:
            self.display_frame(trajectory_vis)
        
        return speed, trajectory_vis, current_traj, desired_traj, self.route_detected

    def display_frame(self, trajectory_vis):
        """Display the trajectory visualization frame."""
        if trajectory_vis is not None:
            self.initialize_window()
            
            # Scale up for better visibility
            height, width = trajectory_vis.shape[:2]
            scale_factor = max(2, 400 // max(height, width))
            new_height = height * scale_factor
            new_width = width * scale_factor
            
            scaled_frame = cv2.resize(trajectory_vis, (new_width, new_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow(self.window_name, scaled_frame)
            cv2.waitKey(1)

    def check_window_closed(self):
        """Check if the display window has been closed."""
        if self.window_initialized:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        return False

    def close_window(self):
        """Close the display window."""
        if self.window_initialized:
            cv2.destroyWindow(self.window_name)
            self.window_initialized = False

    def extract_route_mask(self, roi_array):
        """
        Extract route mask using HSV color filtering with dilation.
        Dilation helps maintain route detection when slightly off-path.
        """
        hsv = cv2.cvtColor(roi_array, cv2.COLOR_RGB2HSV)
        
        # Create mask using HSV thresholds
        mask = cv2.inRange(hsv, self.low_thresh, self.high_thresh)
        
        # Dilate mask to thicken route to maintain route detection when braking late or going slightly off-path
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.mask_dilation_kernel_size, self.mask_dilation_kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=self.mask_dilation_iterations)
        
        return mask

    def find_all_possible_paths(self, skeleton, start_point, max_paths=5):
        """
        Find multiple possible paths through the skeleton using depth-first search.
        """
        height, width = skeleton.shape
        points = np.argwhere(skeleton > 0)
        
        if len(points) == 0:
            return []
        
        # Convert to (x, y) format and create adjacency structure
        skeleton_points = [(int(pt[1]), int(pt[0])) for pt in points]
        point_set = set(skeleton_points)
        
        # Build adjacency graph
        adjacency = {}
        for x, y in skeleton_points:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in point_set:
                        neighbors.append((nx, ny))
            adjacency[(x, y)] = neighbors
        
        all_paths = []
        
        # Depth-first search with backtracking
        def dfs_path_finding(current, path, visited, target_length):
            if len(path) >= target_length or current[1] <= 5:  # Reached top or sufficient length
                return [path.copy()]
            
            paths = []
            current_neighbors = adjacency.get(current, [])

            def neighbor_priority(point):
                px, py = point
                cx, cy = current
                # Prefer upward movement but don't make it exclusive
                y_diff = cy - py
                return (-y_diff * 0.6, abs(px - cx))
            
            sorted_neighbors = sorted(current_neighbors, key=neighbor_priority)
            
            for next_point in sorted_neighbors:
                if next_point not in visited and len(paths) < 3:  # Limit branching
                    new_visited = visited.copy()
                    new_visited.add(next_point)
                    path.append(next_point)
                    
                    sub_paths = dfs_path_finding(next_point, path, new_visited, target_length)
                    paths.extend(sub_paths)
                    
                    path.pop()
            
            return paths
        
        # Find paths of different target lengths to capture various route segments
        target_lengths = [height // 2, height * 2 // 3, height]
        
        for target_length in target_lengths:
            paths = dfs_path_finding(start_point, [start_point], {start_point}, target_length)
            all_paths.extend(paths)
        
        def multi_directional_search():
            # Start multiple searches from different initial directions
            search_paths = []
            
            initial_neighbors = adjacency.get(start_point, [])
            
            for initial_direction in initial_neighbors:
                if len(search_paths) >= max_paths:
                    break
                    
                # Follow this initial direction
                current = initial_direction
                path = [start_point, current]
                visited = {start_point, current}
                
                # Continue following the path
                while len(path) < height * 2 and current[1] > 5:
                    neighbors = [n for n in adjacency.get(current, []) if n not in visited]
                    
                    if not neighbors:
                        break
                    
                    # Choose next point - prefer continuing in a consistent direction
                    if len(path) >= 2:
                        prev_point = path[-2]
                        curr_point = path[-1]
                        
                        # Calculate current direction
                        dir_x = curr_point[0] - prev_point[0]
                        dir_y = curr_point[1] - prev_point[1]
                        
                        # Prefer neighbors that continue in similar direction
                        def direction_continuity(point):
                            new_dir_x = point[0] - curr_point[0]
                            new_dir_y = point[1] - curr_point[1]
                            
                            # Dot product for direction similarity
                            similarity = dir_x * new_dir_x + dir_y * new_dir_y
                            return -similarity  # Negative for max similarity
                        
                        neighbors.sort(key=direction_continuity)
                    
                    next_point = neighbors[0]
                    path.append(next_point)
                    visited.add(next_point)
                    current = next_point
                
                if len(path) >= self.min_segment_length:
                    search_paths.append(path)
            
            return search_paths
        
        multi_paths = multi_directional_search()
        all_paths.extend(multi_paths)
        
        # Filter paths by minimum length
        valid_paths = []
        for path in all_paths:
            if len(path) >= self.min_segment_length:
                path_length = len(path)
                valid_paths.append((path_length, path))
        
        # Sort by length and return top paths
        valid_paths.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in valid_paths[:max_paths]]

    def select_best_path(self, paths):
        """
        Select the best path from multiple candidates based on length.
        """
        if not paths:
            return []
        
        if len(paths) == 1:
            return paths[0]
        
        # Score paths based on length only
        scored_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # Length score (longer is better)
            score = len(path) * 10
            
            scored_paths.append((score, path))
        
        # Return the highest scoring path
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return scored_paths[0][1]

    def extract_centerline(self, mask):
        """
        Extract centerline using skeleton processing.
        Returns ordered list of (x, y) points representing the route centerline.
        """
        binary_mask = (mask > 0).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            return []
        
        height, width = mask.shape
        skeleton = skeletonize(binary_mask).astype(np.uint8)
        
        # Find starting point (closest to bottom center - vehicle position)
        vehicle_x = width // 2 + 8
        vehicle_y = height - 1
        
        skeleton_points = np.argwhere(skeleton > 0)
        if len(skeleton_points) == 0:
            return []
        
        # Convert to (x, y) format and find start point
        points = [(int(pt[1]), int(pt[0])) for pt in skeleton_points]
        start_point = min(points, key=lambda p: (p[0] - vehicle_x)**2 + (p[1] - vehicle_y)**2)
        possible_paths = self.find_all_possible_paths(skeleton, start_point)
        
        if possible_paths:
            best_path = self.select_best_path(possible_paths)
        else:
            best_path = []
        
        return best_path

    def calculate_trajectories(self, roi_array, mask):
        """Calculate current and desired trajectories with path finding."""
        height, width = roi_array.shape[:2]
        
        # Current trajectory: straight line from bottom center to top center
        vehicle_x = width // 2
        current_trajectory = []
        
        for y in range(height - 1, -1, -5):
            current_trajectory.append((vehicle_x, y))
        
        # Desired trajectory: extract centerline from route mask
        desired_trajectory = self.extract_centerline(mask)
        
        return current_trajectory, desired_trajectory

    def analyze_trajectories(self, minimap):
        """Analyze trajectories and return visualization and trajectory data."""
        try:
            mask = self.extract_route_mask(minimap)
            
            if np.sum(mask) == 0:
                # No route detected - return empty trajectories
                return None, [], []
            
            current_traj, desired_traj = self.calculate_trajectories(minimap, mask)
            
            # Create visualization image
            vis_array = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            vis_array[mask > 0] = [255, 255, 255]
            vis_pil = Image.fromarray(vis_array)
            draw = ImageDraw.Draw(vis_pil)
            
            # Draw desired trajectory (GREEN line/points)
            if len(desired_traj) >= 2:
                # Draw as connected line
                for i in range(len(desired_traj) - 1):
                    x1, y1 = desired_traj[i]
                    x2, y2 = desired_traj[i + 1]
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=4)
                
                # Draw points for better visibility
                for x, y in desired_traj[::2]:  # Every 2nd point
                    draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(0, 255, 0))
                    
            elif len(desired_traj) >= 1:
                # Draw as points if we don't have enough for lines
                for x, y in desired_traj:
                    draw.ellipse([(x-4, y-4), (x+4, y+4)], fill=(0, 255, 0))
            
            # Draw current trajectory (RED)
            if len(current_traj) >= 2:
                for i in range(len(current_traj) - 1):
                    x1, y1 = current_traj[i]
                    x2, y2 = current_traj[i + 1]
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 255), width=2)
            elif len(current_traj) >= 1:
                for x, y in current_traj:
                    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 0, 255))
            
            return np.array(vis_pil), current_traj, desired_traj
            
        except Exception as e:
            print(f"Trajectory analysis error: {str(e)}")
            return None, [], []