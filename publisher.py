"""
publisher.py
------------
Simulates two camera feeds by generating random images and publishing
them to an MQTT broker (Mosquitto).

Topics published:
  - camera/object_detection  (Camera 1)
  - camera/lane_detection    (Camera 2)

Topics subscribed (to receive processed results):
  - results/object_detection
  - results/lane_detection
"""

import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ──────────────────────────────────────────
# Broker configuration
# ──────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883

# Publish topics
TOPIC_CAM1 = "camera/object_detection"
TOPIC_CAM2 = "camera/lane_detection"

# Subscribe topics (results coming back from subscribers)
TOPIC_RESULT_OBJ  = "results/object_detection"
TOPIC_RESULT_LANE = "results/lane_detection"

# Image settings
IMG_WIDTH    = 640
IMG_HEIGHT   = 480
SEND_INTERVAL = 1.0   # seconds between each publish cycle


# ──────────────────────────────────────────
# Image generation helpers
# ──────────────────────────────────────────
def generate_random_image(label: str) -> np.ndarray:
    """
    Creates a random-noise image (simulating a raw camera frame)
    and overlays a label and timestamp for identification.
    """
    img = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    cv2.putText(img, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (20, IMG_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1, cv2.LINE_AA)
    return img


def encode_image(img: np.ndarray) -> bytes:
    """Compresses an image to JPEG and returns it as raw bytes."""
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    return buffer.tobytes()


# ──────────────────────────────────────────
# MQTT callbacks
# ──────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Publisher] Connected to broker successfully")
        client.subscribe(TOPIC_RESULT_OBJ)
        client.subscribe(TOPIC_RESULT_LANE)
        print(f"[Publisher] Subscribed to: {TOPIC_RESULT_OBJ}")
        print(f"[Publisher] Subscribed to: {TOPIC_RESULT_LANE}")
    else:
        print(f"[Publisher] Connection failed (rc={rc})")


def on_message(client, userdata, msg):
    """Receives processed images sent back by subscribers and displays them."""
    topic = msg.topic
    print(f"[Publisher] Result received from [{topic}] | {len(msg.payload)} bytes")

    np_arr    = np.frombuffer(msg.payload, dtype=np.uint8)
    result_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if result_img is not None:
        window_name = "Result: Object Detection" if "object" in topic else "Result: Lane Detection"
        cv2.imshow(window_name, result_img)
        cv2.waitKey(1)
    else:
        print(f"[Publisher] Could not decode image from [{topic}]")


def on_disconnect(client, userdata, rc):
    print(f"[Publisher] Disconnected from broker (rc={rc})")


# ──────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────
def main():
    client = mqtt.Client(client_id="publisher_001")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"[Publisher] Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    frame_id = 0
    try:
        while True:
            frame_id += 1

            # --- Camera 1: Object Detection ---
            img1     = generate_random_image(f"CAM1 | Frame #{frame_id}")
            payload1 = encode_image(img1)
            client.publish(TOPIC_CAM1, payload=payload1, qos=1)
            print(f"[Publisher] Published CAM1 -> [{TOPIC_CAM1}] | {len(payload1)} bytes")

            # --- Camera 2: Lane Detection ---
            img2     = generate_random_image(f"CAM2 | Frame #{frame_id}")
            payload2 = encode_image(img2)
            client.publish(TOPIC_CAM2, payload=payload2, qos=1)
            print(f"[Publisher] Published CAM2 -> [{TOPIC_CAM2}] | {len(payload2)} bytes")

            print(f"[Publisher] --- Frame {frame_id} complete ---\n")
            time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Publisher] Stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()