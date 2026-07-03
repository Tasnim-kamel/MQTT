import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ==========================================================
# MQTT Configuration
# ==========================================================
BROKER = "localhost"
PORT = 1883
TOPIC = "camera/raw"

# ==========================================================
# Callback Function
# This function is called whenever a new message arrives.
# ==========================================================
def on_message(client, userdata, msg):

    print("Frame received")

    # Convert MQTT payload into a NumPy array
    image_data = np.frombuffer(msg.payload, dtype=np.uint8)

    # Decode PNG image
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if frame is None:
        print("Failed to decode image")
        return

    # Display the received frame
    cv2.imshow("Received Frame", frame)

    # Refresh the window
    cv2.waitKey(1)


# ==========================================================
# Create MQTT Client
# ==========================================================
client = mqtt.Client()

# Register callback
client.on_message = on_message

# Connect to Broker
client.connect(BROKER, PORT, 60)

# Subscribe to Topic
client.subscribe(TOPIC)

print("Waiting for frames...")

# Start listening forever
client.loop_forever()