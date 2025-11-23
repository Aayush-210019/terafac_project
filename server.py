import asyncio
import json
import websockets
from flask import Flask, request, jsonify
import threading
import aiohttp
app = Flask(__name__)

# --- CORS: allow simple cross-origin calls from control page ---
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

# ---------------------------
# Globals
# ---------------------------
connected = set()
async_loop = None
collision_count = 0  
FLOOR_HALF = 50 

# --- CRITICAL: GLOBAL FOR POSITION ---
latest_position = {"x": 0.0, "y": 0.0, "z": 0.0} 
latest_frame_b64 = None

def corner_to_coords(corner: str, margin=5):
    c = corner.upper()
    x = FLOOR_HALF - margin if "E" in c else -(FLOOR_HALF - margin)
    z = FLOOR_HALF - margin if ("S" in c or "B" in c) else -(FLOOR_HALF - margin)
    if c in ("NE", "EN", "TR"): x, z = (FLOOR_HALF - margin, -(FLOOR_HALF - margin))
    if c in ("NW", "WN", "TL"): x, z = (-(FLOOR_HALF - margin), -(FLOOR_HALF - margin))
    if c in ("SE", "ES", "BR"): x, z = (FLOOR_HALF - margin, (FLOOR_HALF - margin))
    if c in ("SW", "WS", "BL"): x, z = (-(FLOOR_HALF - margin), (FLOOR_HALF - margin))
    return {"x": x, "y": 0, "z": z}

# ---------------------------
# WebSocket Handler
# ---------------------------
async def ws_handler(websocket, path=None):
    global collision_count
    global latest_position
    print("Client connected via WebSocket")
    connected.add(websocket)

    try:
        async for message in websocket:
            print(f"[WS RECEIVED] {message}")
            try:
                data = json.loads(message)
                print("Received raw:", message)

                # --- Simulate positions if simulator doesn't send them ---
                if isinstance(data, dict):
                    if data.get("command") == "move":
                        target = data.get("target", {})
                        if target:
                            # Move instantly to target (for testing)
                            latest_position["x"] = target.get("x", latest_position["x"])
                            latest_position["z"] = target.get("z", latest_position["z"])
                            print(f"[SIMULATION] Updated latest_position to {latest_position}")

                    elif data.get("command") == "move_relative":
                        turn = data.get("turn", 0)
                        distance = data.get("distance", 0)
                        # Simple naive simulation: move in +X,+Z
                        latest_position["x"] += distance * 0.7  # scale factor to simulate direction
                        latest_position["z"] += distance * 0.7
                        print(f"[SIMULATION] Updated latest_position to {latest_position}")

            except Exception as e:
                print("Error parsing WS message:", e)
                continue

            try:
                # --- Save position if simulator sends it ---
                if isinstance(data, dict) and "position" in data:
                    latest_position = data["position"]

                # --- Forward captured image to /frame endpoint ---
                if isinstance(data, dict) and data.get("type") == "capture_image_response":
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            "http://localhost:5000/frame",
                            json={"image": data.get("image")}
                        )

                # --- Track collisions ---
                if isinstance(data, dict) and data.get("type") == "collision" and data.get("collision"):
                    collision_count += 1

            except Exception as e:
                print("Error in ws handler post-processing:", e)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

    finally:
        connected.remove(websocket)

def broadcast(msg: dict):
    if not connected:
        return False
    for ws in list(connected):
        asyncio.run_coroutine_threadsafe(ws.send(json.dumps(msg)), async_loop)
    return True

# ---------------------------
# NEW ENDPOINT: Get Robot Position
# ---------------------------
@app.route('/position', methods=['GET'])
def get_position():
    """Client fetches the latest robot position."""
    global latest_position
    return jsonify({"position": latest_position})

# ---------------------------
# MOVEMENT AND CAPTURE ENDPOINTS 
# ---------------------------
@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    if not data or 'x' not in data or 'z' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "x" and "z".'}), 400
    x, z = data['x'], data['z']
    msg = {"command": "move", "target": {"x": x, "y": 0, "z": z}}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move command sent', 'command': msg})

@app.route('/move_rel', methods=['POST'])
def move_rel():
    data = request.get_json()
    if not data or 'turn' not in data or 'distance' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "turn" and "distance".'}), 400
    msg = {"command": "move_relative", "turn": data['turn'], "distance": data['distance']}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move relative command sent', 'command': msg})

@app.route('/stop', methods=['POST'])
def stop():
    msg = {"command": "stop"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'stop command sent', 'command': msg})

@app.route('/capture', methods=['POST'])
def capture():
    msg = {"command": "capture_image"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'capture command sent', 'command': msg})

# ---------------- FRAME ENDPOINTS -------------------

@app.route('/frame', methods=['POST'])
def receive_frame():
    """Simulator POSTs the Base64 frame here."""
    global latest_frame_b64

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' key"}), 400

    latest_frame_b64 = data['image']
    # If successful, this is where the client *would* see "Frame received" in the console.
    return jsonify({"status": "ok", "message": "frame received"})


@app.route('/frame', methods=['GET'])
def get_frame():
    """Client fetches the latest frame."""
    global latest_frame_b64

    if latest_frame_b64 is None:
        return jsonify({"error": "No frame available"}), 500

    return jsonify({"image": latest_frame_b64})

# ---------------- OTHER ENDPOINTS -------------------
@app.route('/goal', methods=['POST'])
def set_goal():
    data = request.get_json() or {}
    if 'corner' in data:
        pos = corner_to_coords(str(data['corner']))
    elif 'x' in data and 'z' in data:
        pos = {"x": float(data['x']), "y": float(data.get('y', 0)), "z": float(data['z'])}
    else:
        return jsonify({'error': 'Provide {"corner":"NE|NW|SE|SW"} OR {"x":..,"z":..}'}), 400

    msg = {"command": "set_goal", "position": pos}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'goal set', 'goal': pos})

@app.route('/collisions', methods=['GET'])
def get_collisions():
    global collision_count
    return jsonify({'count': collision_count})

@app.route('/reset', methods=['POST'])
def reset():
    global collision_count
    collision_count = 0
    global latest_position
    latest_position = {"x": 0.0, "y": 0.0, "z": 0.0} # Reset position on server side too
    if not broadcast({"command": "reset"}):
        return jsonify({'status': 'reset done (no simulators connected)', 'collisions': collision_count})
    return jsonify({'status': 'reset broadcast', 'collisions': collision_count})


# ---------------------------
# Flask Thread
# ---------------------------
def start_flask():
    app.run(port=5000, threaded=True)

# ---------------------------
# Main Async for WebSocket
# ---------------------------
async def main():
    global async_loop
    async_loop = asyncio.get_running_loop()
    ws_server = await websockets.serve(ws_handler, "localhost", 8080)
    print("WebSocket server started on ws://localhost:8080")
    await ws_server.wait_closed()

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(main())