import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime  # Import the datetime module

# Set up the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variable to store the image path after capturing
image_path = 'image.jpg'
text_file_path = 'detected_plates.txt'  # Path to save the text file

while True:
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)

    # Preprocessing to enhance image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 30, 200)
    cv2.imshow('Edged', edged)

    # Finding contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)
        cv2.imshow('Plate Contour', frame)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        (x, y, w, h) = cv2.boundingRect(screenCnt)
        cropped = frame[y:y+h, x:x+w]
        cv2.imshow('Cropped Plate', cropped)

        # OCR
        text = pytesseract.image_to_string(cropped, config='--psm 11 --oem 3')
        license_plate = re.sub(r'[\W_]+', '', text)  # Remove unwanted characters using a raw string
        if re.match(r'^[A-Z0-9]{1,8}$', license_plate):  # Validate plate format
            cv2.imwrite(image_path, frame)
            print(f"Detected Plate: {license_plate}")
            print(f"Image saved as '{image_path}'")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date and time

            # Save to text file with time stamp
            with open(text_file_path, 'a') as file:
                file.write(f"{current_time} - Detected Plate: {license_plate}\n")

            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# from pygrabber.dshow_graph import FilterGraph

# graph = FilterGraph()
# print(graph.get_input_devices())  


# pip install torch-2.4.0+cu118-cp312-cp312-win_amd64.whl
# pip install torchvision-0.16.0+cu118-cp312-cp312-win_amd64.whl
# pip install torchaudio-2.4.0+cu118-cp312-cp312-win_amd64.whl
