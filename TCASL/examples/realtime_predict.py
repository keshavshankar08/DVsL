import cv2
from collections import deque
from tcasl import TCASL

def main():
    # Initialize class
    tcasl = TCASL()
    
    # Setup video capture
    cap = cv2.VideoCapture(0)

    # Buffer for predictions
    prediction_buffer = deque(maxlen=15)
    
    # Initial frame
    previous_frame = None

    while True:
        # Fetch frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip and convert frame to gray
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocess frame
        processed_frame = tcasl.preprocess_frame(gray_frame)
        
        # Skip first iteration
        if previous_frame is None:
            previous_frame = processed_frame
            continue
            
        # Compute temporal contrast
        tc_frame = tcasl.compute_temporal_contrast(previous_frame, processed_frame, threshold=20)
        
        # Predict signed letter
        prediction = tcasl.predict(tc_frame)

        # Add prediction to buffer
        prediction_buffer.append(prediction)
        
        # Get majority vote prediction from buffer
        majority_pred = max(set(prediction_buffer), key=prediction_buffer.count)
        
        # Show result
        processed_frame_copy = processed_frame.copy()
        cv2.putText(processed_frame_copy, majority_pred.upper(), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Grayscale Frame", processed_frame_copy)
        cv2.imshow("Temporal Contrast Frame", tc_frame)
        
        # Update previous frame as current
        previous_frame = processed_frame
        
        # Quit if 'q' pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    # Release video capture
    cap.release()

    # Destroy windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()