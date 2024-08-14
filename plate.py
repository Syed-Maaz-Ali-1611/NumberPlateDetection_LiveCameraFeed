import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os

def save_license_plate(text):
    file_path = 'license_plates.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(text + '\n')
    else:
        with open(file_path, 'a') as file:
            file.write(text + '\n')
    
    with open(file_path, 'r') as file:
        print("Current License Plates in File:")
        print(file.read())

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title('Grayscale Image')
    plt.show()

    # Apply bilateral filter to reduce noise
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detect edges using Canny edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Display the edge-detected image
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.title('Edge Detection')
    plt.show()

    # Find contours in the edged image
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize the location variable
    location = None

    # Loop through the contours to find the license plate contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # Check if location is None
    if location is None:
        print("No contour with 4 vertices found for image:", image_path)
        return

    # Create a mask for the detected license plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Display the masked license plate
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title('Masked License Plate')
    plt.show()

    # Find the coordinates of the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    # Crop the license plate from the image
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Display the cropped license plate
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Cropped License Plate')
    plt.show()

    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])

    # Read the text from the cropped license plate
    result = reader.readtext(cropped_image)

    if not result:
        print("No text detected in the cropped license plate for image:", image_path)
        return

    # Extract the text from the OCR result
    text = result[0][-2]

    # Save the license plate text to file
    save_license_plate(text)

    # Define the font for the text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw the text and a rectangle around the license plate on the original image
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

    # Display the final image with the detected license plate and text
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title('Final Image with Detected License Plate')
    plt.show()

# Process the provided image
process_image('image1.jpg')