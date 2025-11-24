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
GOAL_POSITION = {"x": 450.0, "z": 450.0}
GOAL_THRESHOLD = 3.0

NEON_GREEN_HSV_LOWER = np.array([40, 150, 150])
NEON_GREEN_HSV_UPPER = np.array([80, 255, 255])

OBSTACLE_PIXEL_THRESHOLD = 200
COLLISION_PIXEL_THRESHOLD = 5000
NEAR_OBSTACLE_PIXELS = 1500
GO_THRESHOLD_PIXELS = 50

current_state = GO_TO_GOAL
def broadcast_goal_flag():
    try:
        requests.post(f"{FLASK_API_URL}/goal", json=GOAL_POSITION)
        print(f"[INIT] Goal flag broadcasted → ({GOAL_POSITION['x']}, {GOAL_POSITION['z']})")
    except Exception as e:
        print("[INIT ERROR] Could not broadcast goal:", e)


def post_command(endpoint, payload=None):
    try:
        requests.post(f"{FLASK_API_URL}/{endpoint}", json=payload, timeout=0.1)
    except:
        pass


def normalize_angle(a):
    return a % 360


def move_rel(turn_angle, dist, pos=None, heading=None):
    post_command("move_rel", {"turn": turn_angle, "distance": dist})
    if pos is not None and heading is not None:
        heading = normalize_angle(heading + turn_angle)
        r = math.radians(heading)
        pos["x"] += dist * math.cos(r)
        pos["z"] += dist * math.sin(r)
        return heading
    return heading


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
        if resp.status_code != 200:
            return None
        b64 = resp.json().get("image", "").replace("data:image/png;base64,", "")
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None


def analyze_vision_relaxed(frame):
    if frame is None:
        return {'zones': {'center': False}, 'collision': False, 'center_px': 0}

    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, NEON_GREEN_HSV_LOWER, NEON_GREEN_HSV_UPPER)

    roi = mask[int(h * 0.5):h, :]
    c1, c2 = int(w * 0.45), int(w * 0.55)
    center = roi[:, c1:c2]

    px = cv2.countNonZero(center)
    area = center.shape[0] * center.shape[1]
    frac = px / area

    hit = frac > 0.2
    zones = {'center': hit}

    dbg = frame.copy()
    cv2.line(dbg, (c1, 0), (c1, h), (255, 0, 0), 2)
    cv2.line(dbg, (c2, 0), (c2, h), (255, 0, 0), 2)

    if hit:
        cv2.putText(dbg, "OBSTACLE", (c1, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Vision", dbg)
    cv2.waitKey(1)

    return {'zones': zones, 'collision': hit, 'center_px': px}


def main_loop():
    global current_state

    print("--- AGENT STARTED ---")
    post_command("reset")
    sleep(1)
    broadcast_goal_flag()

    pos = {"x": 0.0, "z": 0.0}
    heading = 0.0
    current_state = GO_TO_GOAL
    hit_dist = float('inf')

    step = 0
    wall_enter_step = None

    while step < 1000:
        step += 1
        trigger_capture()
        sleep(0.1)

        frame = get_frame()
        vis = analyze_vision_relaxed(frame)

        center_px = vis['center_px']
        collision = vis['collision']
        zones = vis['zones']

        dist = distance_to_goal(pos)
        mode = "GO" if current_state == GO_TO_GOAL else "FOLLOW"
        print(f"Step {step} | {mode} | Pos ({pos['x']:.1f},{pos['z']:.1f}) "
              f"| Dist {dist:.1f} | Coll {collision} | Center {center_px}")

        if dist < GOAL_THRESHOLD:
            print("GOAL REACHED!")
            break

        if current_state == GO_TO_GOAL:
            if collision:
                print("[LOG] COLLISION → WALL FOLLOW")
                current_state = WALL_FOLLOW
                hit_dist = dist
                wall_enter_step = step
                heading = move_rel(30, -2.0, pos, heading)
            else:
                tgt = get_heading_to_goal(pos, GOAL_POSITION)
                d = (tgt - heading + 180) % 360 - 180
                heading = move_rel(d, STEP_DISTANCE, pos, heading)

        else:
            TH = 3000

            if center_px > TH:
                heading = move_rel(40, STEP_DISTANCE, pos, heading)
            else:
                heading = move_rel(5, STEP_DISTANCE, pos, heading)

            tgt = get_heading_to_goal(pos, GOAL_POSITION)
            d = (tgt - heading + 180) % 360 - 180

            clear = center_px < TH
            goal_front = abs(d) < 70
            good = dist < hit_dist * 1.05

            if clear and goal_front and good:
                print("[LOG] CLEAR → GO_TO_GOAL")
                current_state = GO_TO_GOAL
                wall_enter_step = None

            if wall_enter_step is not None and step - wall_enter_step > 40:
                print("[LOG] WALL FOLLOW TIMEOUT → GO_TO_GOAL")
                current_state = GO_TO_GOAL
                wall_enter_step = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
