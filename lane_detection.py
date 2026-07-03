"""
subscriber_lane_detection.py
-----------------------------
Subscribes to: camera/lane_detection
Simulates lane detection by drawing lane lines and a drivable-area polygon
on the received image.
Publishes the processed image to: results/lane_detection

In a real system, replace simulate_lane_detection() with an actual
lane-detection algorithm (e.g. Hough Lines, SCNN, LaneNet).
"""

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ──────────────────────────────────────────
# Broker configuration
# ──────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883

TOPIC_SUBSCRIBE = "camera/lane_detection"
TOPIC_PUBLISH   = "results/lane_detection"


# ──────────────────────────────────────────
# Processing: simulated lane detection
# ──────────────────────────────────────────
def simulate_lane_detection(img: np.ndarray) -> np.ndarray:
    """
    Draws left and right lane boundaries, a semi-transparent drivable-area
    polygon, a dashed center line, and a lateral deviation readout
    to simulate a lane detection pipeline.

    Replace this function with a real model for production use.
    """
    result = img.copy()
    h, w   = result.shape[:2]

    # Vanishing point (slightly randomized around the image center)
    vp_x = w // 2 + np.random.randint(-30, 30)
    vp_y = int(h * 0.45) + np.random.randint(-20, 20)

    # --- Left lane boundary ---
    left_bottom_x = np.random.randint(0, w // 4)
    left_pts = np.array([
        [left_bottom_x,       h],
        [left_bottom_x + 40,  int(h * 0.75)],
        [vp_x - 30,           int(h * 0.60)],
        [vp_x,                vp_y],
    ], dtype=np.int32)
    cv2.polylines(result, [left_pts], isClosed=False,
                  color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    # --- Right lane boundary ---
    right_bottom_x = np.random.randint(3 * w // 4, w)
    right_pts = np.array([
        [right_bottom_x,      h],
        [right_bottom_x - 40, int(h * 0.75)],
        [vp_x + 30,           int(h * 0.60)],
        [vp_x,                vp_y],
    ], dtype=np.int32)
    cv2.polylines(result, [right_pts], isClosed=False,
                  color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    # --- Drivable-area polygon (semi-transparent green fill) ---
    lane_poly = np.array([
        left_pts[0], left_pts[1], left_pts[2], left_pts[3],
        right_pts[3], right_pts[2], right_pts[1], right_pts[0],
    ], dtype=np.int32)
    overlay = result.copy()
    cv2.fillPoly(overlay, [lane_poly], color=(0, 200, 0))
    cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)

    # --- Dashed center line ---
    center_pts = [
        ((left_pts[i][0] + right_pts[i][0]) // 2,
         (left_pts[i][1] + right_pts[i][1]) // 2)
        for i in range(len(left_pts))
    ]
    for i in range(len(center_pts) - 1):
        if i % 2 == 0:   # draw every other segment to create a dashed effect
            cv2.line(result, center_pts[i], center_pts[i + 1],
                     (255, 255, 0), 2, cv2.LINE_AA)

    # --- Lateral deviation readout ---
    deviation   = round(np.random.uniform(-0.3, 0.3), 2)
    dev_text    = f"Deviation: {deviation:+.2f} m"
    dev_color   = (0, 255, 0) if abs(deviation) < 0.15 else (0, 100, 255)
    cv2.putText(result, dev_text, (w // 2 - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dev_color, 2, cv2.LINE_AA)

    # Subscriber watermark
    cv2.putText(result, "[ Lane Detection ]", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)

    return result


def encode_image(img: np.ndarray) -> bytes:
    """Encodes an OpenCV image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode image")
    return buffer.tobytes()


# ──────────────────────────────────────────
# MQTT callbacks
# ──────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[LD Subscriber] Connected to broker")
        client.subscribe(TOPIC_SUBSCRIBE, qos=1)
        print(f"[LD Subscriber] Subscribed to: {TOPIC_SUBSCRIBE}")
    else:
        print(f"[LD Subscriber] Connection failed (rc={rc})")


def on_message(client, userdata, msg):
    """Handles an incoming raw image, processes it, and publishes the result."""
    print(f"[LD Subscriber] Image received | {len(msg.payload)} bytes")

    # Decode incoming JPEG bytes
    np_arr = np.frombuffer(msg.payload, dtype=np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("[LD Subscriber] Could not decode image — skipping")
        return

    # Run simulated lane detection
    result_img = simulate_lane_detection(img)
    print("[LD Subscriber] Lane lines drawn")

    # Display locally
    cv2.imshow("Lane Detection - Live", result_img)
    cv2.waitKey(1)

    # Publish result back to the publisher
    result_bytes = encode_image(result_img)
    client.publish(TOPIC_PUBLISH, payload=result_bytes, qos=1)
    print(f"[LD Subscriber] Result published -> [{TOPIC_PUBLISH}] | {len(result_bytes)} bytes\n")


def on_disconnect(client, userdata, rc):
    print(f"[LD Subscriber] Disconnected (rc={rc})")


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────
def main():
    client = mqtt.Client(client_id="subscriber_lane_detection")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"[LD Subscriber] Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[LD Subscriber] Stopped by user")
    finally:
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()