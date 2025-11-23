import requests
import base64
import numpy as np
import cv2
from time import sleep
import math

FLASK_API_URL = "http://localhost:5000"

GO_TO_GOAL = 1
WALL_FOLLOW = 2

STEP_DISTANCE = 5.0
TURN_ANGLE = 10
GOAL_POSITION = {"x": 45.0, "z": 45.0}
GOAL_THRESHOLD = 3.0

NEON_GREEN_HSV_LOWER = np.array([40, 150, 150])
NEON_GREEN_HSV_UPPER = np.array([80, 255, 255])

OBSTACLE_PIXEL_THRESHOLD = 200
COLLISION_PIXEL_THRESHOLD = 5000
NEAR_OBSTACLE_PIXELS = 1500   # When to start wall-following
GO_THRESHOLD_PIXELS = 50       # Path is clear if less than this

current_state = GO_TO_GOAL

def post_command(endpoint, payload=None):
    try:
        requests.post(f"{FLASK_API_URL}/{endpoint}", json=payload, timeout=0.1)
    except: 
        pass

def normalize_angle(angle):
    return angle % 360

def move_rel(turn_angle, distance, pos=None, heading_deg=None):
    post_command("move_rel", {"turn": turn_angle, "distance": distance})
    if pos is not None and heading_deg is not None:
        new_heading = normalize_angle(heading_deg + turn_angle)
        rad = math.radians(new_heading)
        pos["x"] += distance * math.cos(rad)
        pos["z"] += distance * math.sin(rad)
        return new_heading
    return heading_deg

def get_heading_to_goal(pos, goal):
    dx = goal["x"] - pos["x"]
    dz = goal["z"] - pos["z"]
    return normalize_angle(math.degrees(math.atan2(dz, dx)))

def distance_to_goal(pos, goal=GOAL_POSITION):
    return math.sqrt((goal["x"] - pos["x"])**2 + (goal["z"] - pos["z"])**2)

def trigger_capture():
    post_command("capture")

def get_frame():
    try:
        resp = requests.get(f"{FLASK_API_URL}/frame", timeout=1)
        if resp.status_code != 200: return None
        data = resp.json()
        img_b64 = data.get("image", "").replace("data:image/png;base64,", "")
        return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    except: 
        return None

def analyze_vision_relaxed(frame):
    if frame is None: 
        return {'zones': {'center': False}, 'collision': False, 'center_px': 0}

    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, NEON_GREEN_HSV_LOWER, NEON_GREEN_HSV_UPPER)
    roi = mask[int(h*0.5):h, :]  # lower half

    center_start = int(w * 0.45)
    center_end = int(w * 0.55)
    center_roi = roi[:, center_start:center_end]

    center_px = cv2.countNonZero(center_roi)

# Compute how full the window is (0.0 to 1.0)
    center_area = center_roi.shape[0] * center_roi.shape[1]
    green_fraction = center_px / center_area

# New obstacle rules (MUCH more stable)
    is_obstacle = green_fraction > 0.2     #parameters can be changed  
    collision = green_fraction > 0.2        

    zones = {'center': is_obstacle}


    # Debug visual
    debug = frame.copy()
    cv2.line(debug, (center_start, 0), (center_start, h), (255, 0, 0), 2)
    cv2.line(debug, (center_end, 0), (center_end, h), (255, 0, 0), 2)
    if zones['center']: 
        cv2.putText(debug, "NEAR OBSTACLE", (center_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("Tunnel Vision", debug)
    cv2.waitKey(1)

    return {'zones': zones, 'collision': collision, 'center_px': center_px}

def main_loop():
    global current_state
    print("--- AGGRESSIVE BUG 2 AGENT STARTED ---")
    post_command("reset")
    sleep(1)

    current_state = GO_TO_GOAL
    pos = {"x": 0.0, "z": 0.0}
    heading = 0.0
    hit_distance = float('inf')

    step = 0
    wall_follow_start_step = None   
    while step < 1000:
        step += 1
        trigger_capture()
        sleep(0.1)
        frame = get_frame()
        data = analyze_vision_relaxed(frame)

        zones = data['zones']
        collision = data['collision']
        center_px = data['center_px']
        dist = distance_to_goal(pos)

        state_str = "GO_TO" if current_state == GO_TO_GOAL else "FOLLOW"
        print(f"Step {step} | {state_str} | Pos: ({pos['x']:.1f},{pos['z']:.1f}) | Dist: {dist:.1f} | Coll: {collision} | CenterPX: {center_px}")

        if dist < GOAL_THRESHOLD:
            print("GOAL REACHED!")
            break

        if current_state == GO_TO_GOAL:
            if collision:
                print("[LOG] COLLISION! Switching to WALL_FOLLOW")
                current_state = WALL_FOLLOW
                hit_distance = dist
                wall_follow_start_step = step    # <<<  entry step
                heading = move_rel(45, -2.0, pos, heading)
            elif center_px > NEAR_OBSTACLE_PIXELS:
                print("[LOG] Near obstacle! Switching to WALL_FOLLOW")
                current_state = WALL_FOLLOW
                wall_follow_start_step = step 
                hit_distance = dist
            else:
                # Drive straight to goal
                tgt = get_heading_to_goal(pos, GOAL_POSITION)
                diff = (tgt - heading + 180) % 360 - 180
                turn = max(min(diff, TURN_ANGLE), -TURN_ANGLE)
                heading = move_rel(turn, STEP_DISTANCE, pos, heading)

        elif current_state == WALL_FOLLOW:
    
            # 1. Follow wall but bias outward
            if zones['center']:
                heading = move_rel(40, STEP_DISTANCE, pos, heading)
            else:
                heading = move_rel(5, STEP_DISTANCE, pos, heading)

            # 2. Check if we can return to GO_TO_GOAL
            tgt_heading = get_heading_to_goal(pos, GOAL_POSITION)
            heading_diff = (tgt_heading - heading + 180) % 360 - 180

            path_clear = center_px < 1200
            goal_ahead = abs(heading_diff) < 70
            closer = dist < hit_distance * 1.05

            if path_clear and goal_ahead and closer:
                print("[LOG] Tunnel clear! Switching to GO_TO_GOAL")
                current_state = GO_TO_GOAL
                wall_follow_start_step = None   
            # 3. Safety escape timeout
            # 3. Safety escape timeout (only if still in wall-follow)
            if current_state == WALL_FOLLOW and wall_follow_start_step is not None:
                if step - wall_follow_start_step > 40:
                    print("[LOG] Forced exit from wall-follow (timeout)")
                    current_state = GO_TO_GOAL
                    wall_follow_start_step = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
