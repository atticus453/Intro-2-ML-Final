import cv2
import numpy as np

# Load your pre-trained model

# Define a function to preprocess frames
def preprocess_frame(frame, target_size):
    """
    Resize and normalize the frame for the model.
    """
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0  # Normalize to [0,1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Define labels for the emotions
labels = ['Happy', 'Sad', 'Neutral']  # Replace with your actual labels

# Initialize the webcam
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()  # Read a frame from the webcam
    if not ret:
        break
    
    # Preprocess the frame
    input_frame = preprocess_frame(frame, (32, 32))  # Adjust to your model's input size
    
    # Make a prediction
    # predictions = model.predict(input_frame)
    # predicted_label = labels[np.argmax(predictions)]  # Get the label with the highest probability
    predicted_label = labels[1]
    # Display the prediction on the frame
    cv2.putText(frame, f'Emotion: {predicted_label}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Dog Emotion Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
camera.release()
cv2.destroyAllWindows()