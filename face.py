import cv2

# Create a face cascade object

face_cascade = cv2.CascadeClassifier("/haarcascade_frontalface_default.xml")

# Open the camera

camera = cv2.VideoCapture(0)

# Loop until the user presses "q" key

while True:
  
    # Get the image from the camera
    
    ret, image = camera.read()
    
    # Convert the image to grayscale
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around the faces
    
    for (x, y, w, h) in faces:
      
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    # Show the image on the screen
    
    cv2.imshow("Face Detection", image)
    
    # Wait for a key press
    
    if cv2.waitKey(30) & 0xFF == ord("q"):
      
        break

# Release the camera and close the window

camera.release()

cv2.destroyAllWindows()
