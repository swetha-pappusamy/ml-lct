import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the pre-trained model (you can replace it with your model)
model = tf.keras.models.load_model('hand_gesture_model.h5')  # Replace with your model path

# Class labels for the gestures (this should match your model's output classes)
class_labels = ['Gesture 1', 'Gesture 2', 'Gesture 3', 'Gesture 4', 'Gesture 5']

# Initialize MediaPipe Hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Flip the frame horizontally for a more natural selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe needs RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract the landmarks (22 points for each hand)
            hand_points = []
            for lm in landmarks.landmark:
                hand_points.append([lm.x, lm.y, lm.z])
            
            # Convert to numpy array
            hand_points = np.array(hand_points)

            # Process the hand landmarks (e.g., normalization or reshaping for the model)
            # Assuming the model expects an array of shape (1, 22, 3) for each hand
            hand_input = np.expand_dims(hand_points, axis=0)

            # Predict the gesture using the model
            predictions = model.predict(hand_input)
            predicted_class = np.argmax(predictions)

            # Display the predicted class on the frame
            cv2.putText(frame, f"Predicted: {class_labels[predicted_class]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
