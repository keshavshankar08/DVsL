import cv2
from tcasl import TCASL

def main():
    # Initialize class
    tcasl = TCASL()
    
    # Paths to example frames
    path1 = "frame_1.png"
    path2 = "frame_2.png"
    
    # Read frames
    frame1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess frames
    processed1 = tcasl.preprocess_frame(frame1)
    processed2 = tcasl.preprocess_frame(frame2)
    
    # Compute temporal contrast
    tc_frame = tcasl.compute_temporal_contrast(processed1, processed2)

    # Predict signed letter
    prediction = tcasl.predict(tc_frame)

    # Show result
    print(f"Predicted ASL Character: {prediction.upper()}")

if __name__ == "__main__":
    main()