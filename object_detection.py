"""
subscriber_object_detection.py
-------------------------------
Subscribes to: camera/object_detection
Simulates object detection by drawing random bounding boxes on the received image.
Publishes the processed image to: results/object_detection

In a real system, replace simulate_object_detection() with an actual
inference call (e.g. YOLOv8, SSD, Faster R-CNN).
"""

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ──────────────────────────────────────────
# Broker configuration
# ──────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883

TOPIC_SUBSCRIBE = "camera/object_detection"
TOPIC_PUBLISH   = "results/object_detection"

# Fake class labels and their corresponding BGR colors
FAKE_LABELS = ["Car", "Person", "Truck", "Bus", "Motorcycle", "Bicycle"]
LABEL_COLORS = [
    (0,   255,   0),   # green
    (255,   0,   0),   # blue
    (0,     0, 255),   # red
    (255, 255,   0),   # cyan
    (0,   255, 255),   # yellow
    (255,   0, 255),   # magenta
]


# ──────────────────────────────────────────
# Processing: simulated object detection
# ──────────────────────────────────────────
def simulate_object_detection(img: np.ndarray) -> np.ndarray:
    """
    Draws random bounding boxes with labels and confidence scores
    to simulate an object detection pipeline.

    Replace this function with a real model for production use.
    """
    result   = img.copy()
    h, w     = result.shape[:2]
    num_boxes = np.random.randint(2, 6)

    for _ in range(num_boxes):
        # Random box coordinates
        x1 = np.random.randint(0, w - 80)
        y1 = np.random.randint(0, h - 60)
        x2 = np.random.randint(x1 + 40, min(x1 + 200, w))
        y2 = np.random.randint(y1 + 30, min(y1 + 150, h))

        idx   = np.random.randint(0, len(FAKE_LABELS))
        label = FAKE_LABELS[idx]
        color = LABEL_COLORS[idx]
        conf  = round(np.random.uniform(0.50, 0.99), 2)

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(result, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Subscriber watermark
    cv2.putText(result, "[ Object Detection ]", (10, result.shape[0] - 10),
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
        print("[OD Subscriber] Connected to broker")
        client.subscribe(TOPIC_SUBSCRIBE, qos=1)
        print(f"[OD Subscriber] Subscribed to: {TOPIC_SUBSCRIBE}")
    else:
        print(f"[OD Subscriber] Connection failed (rc={rc})")


def on_message(client, userdata, msg):
    """Handles an incoming raw image, processes it, and publishes the result."""
    print(f"[OD Subscriber] Image received | {len(msg.payload)} bytes")

    # Decode incoming JPEG bytes
    np_arr = np.frombuffer(msg.payload, dtype=np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("[OD Subscriber] Could not decode image — skipping")
        return

    # Run simulated detection
    result_img = simulate_object_detection(img)
    print("[OD Subscriber] Bounding boxes drawn")

    # Display locally
    cv2.imshow("Object Detection - Live", result_img)
    cv2.waitKey(1)

    # Publish result back to the publisher
    result_bytes = encode_image(result_img)
    client.publish(TOPIC_PUBLISH, payload=result_bytes, qos=1)
    print(f"[OD Subscriber] Result published -> [{TOPIC_PUBLISH}] | {len(result_bytes)} bytes\n")


def on_disconnect(client, userdata, rc):
    print(f"[OD Subscriber] Disconnected (rc={rc})")


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────
def main():
    client = mqtt.Client(client_id="subscriber_object_detection")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"[OD Subscriber] Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[OD Subscriber] Stopped by user")
    finally:
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()