import cv2
import os
import subprocess

# Yakeeni banayein ke dataset/images folder mojood ho
os.makedirs("dataset/images", exist_ok=True)

# Laptop ka default webcam open karein
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam open nahi ho saka!")
    exit()

print("==================================================")
print("Webcam shuru ho gaya hai!")
print("Image capture karne ke liye keyboard par 's' dabayein.")
print("Band karne ke liye 'q' dabayein.")
print("==================================================")

captured_image_path = "dataset/images/captured_test.jpg"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame read nahi ho saka!")
        break

    # Live preview dikhayein
    cv2.imshow("Webcam Measurement - Press 's' to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(captured_image_path, frame)
        print(f"\n[SUCCESS] Image save ho gayi hai: {captured_image_path}")
        break
    elif key == ord('q'):
        print("Cancelled by user.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Image capture hone ke foran baad measurement script run kar dein
print("\nMeasurement start ki ja rahi hai...")
subprocess.run([
    "python", "inference/measure.py",
    "--image", captured_image_path,
    "--calib", "calibration/camera_params.json",
    "--weights", "models/maskrcnn_best.pth"
])