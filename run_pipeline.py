import time
import win32api
import dxcam
from segmentation import Segmentation
from control import Control
from image_processing import ImageProcessing
from gui import GUI
from SpeedAcquisition.digit_acquisition import move

"""
The pipeline integrates trajectory-based navigation with semantic segmentation safety overrides.
Uses minimap route analysis for primary navigation decisions while employing segmentation to detect and avoid hazards.
"""

ROUTE_LOST_FRAME_THRESHOLD = 3
# Number of consecutive frames without a detected route before the pipeline pauses.

if __name__ == "__main__":
    # move("RAGE Multiplayer")
    move("Grand Theft Auto V")
    roi = (0, 31, 1152, 894)
    camera = dxcam.create(output_idx=0)
    camera.start(region=roi, target_fps=30, video_mode=True)
    
    control = Control(2)
    image_processing = ImageProcessing()
    segmentation = Segmentation()
    gui = GUI()

    pause = False
    key_pressed = False
    segmentation_key_pressed = False
    segmentation_visualization = True
    collision_warnings = []
    segmentation_action = None
    route_lost_counter = 0

    while True:
        start = time.time()
        
        # Check for exit (Home key) or GUI close
        if win32api.GetAsyncKeyState(0x24) & 0x8001 > 0 or not gui.run:
            control.reset()
            break

        # Check if image processing window was closed
        if image_processing.check_window_closed():
            control.reset()
            break

        # Check for pause (End key)
        if win32api.GetAsyncKeyState(0x23) & 0x8001 > 0:
            if not key_pressed:
                if not pause:
                    pause = True
                    control.reset()
                    print("\nPause!")
                else:
                    pause = False
                key_pressed = True
        else:
            key_pressed = False

        if pause:
            gui.update(0.5, 0, 0, 0, 0, None, is_paused=True)
            time.sleep(0.1)
            continue

        # Check for segmentation toggle
        if win32api.GetAsyncKeyState(0xDC) & 0x8001 > 0:
            if not segmentation_key_pressed:
                segmentation.toggle_segmentation(segmentation_visualization)
                segmentation_key_pressed = True
        else:
            segmentation_key_pressed = False

        # Capture and process frame
        image = camera.get_latest_frame()
        speed, trajectory_vis, current_traj, desired_traj, route_detected = image_processing.process_image(image)

        # Track consecutive frames without a route before deciding to pause.
        if not route_detected:
            route_lost_counter += 1
        else:
            route_lost_counter = 0

        if route_lost_counter >= ROUTE_LOST_FRAME_THRESHOLD:
            if not pause:
                pause = True
                control.reset()
                print(
                    f"\nNo route detected for {route_lost_counter} consecutive frame(s) "
                    f"(threshold: {ROUTE_LOST_FRAME_THRESHOLD}) - Paused! "
                    "Press End to resume when route is available."
                )
            gui.update(0.5, 0, 0, speed, 1 / (time.time() - start), None, is_paused=True)
            time.sleep(0.1)
            continue
        
        # If we are within the grace window (counter > 0 but below threshold), skip
        # navigation updates for this frame to avoid acting on stale trajectories.
        if route_lost_counter > 0:
            gui.update(0.5, 0, 0, speed, 1 / (time.time() - start), None, is_paused=False)
            time.sleep(0.1)
            continue

        # Process segmentation for hazard detection and safety overrides
        collision_warnings = []
        segmentation_action = None
        if segmentation.segmentation_active:
            _, collision_warnings, segmentation_action = segmentation.process_image(image, speed)
        
        # Apply navigation controls with segmentation safety overrides
        control.apply_navigation_controls(
            current_traj, 
            desired_traj, 
            speed, 
            collision_warnings, 
            segmentation_action
        )
        
        # Update GUI with segmentation action info
        fps = 1 / (time.time() - start)
        gui.update(
            control.steering, 
            control.acceleration, 
            control.brake, 
            speed, 
            fps, 
            segmentation_action,
            is_paused=False
        )

    image_processing.close_window()
    camera.stop()