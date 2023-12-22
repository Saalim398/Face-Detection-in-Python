#face detection system in python using cv2 module 

import cv2


image = r"image.jpg"  #image path


img = cv2.imread(image)
#converting color image to gray image 
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cascade classifier is used to detect faces
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#detectmultiscale is amethid for detecting different size faces
face = faceClassifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)



window_name = 'Detected face'

cv2.imshow(window_name,img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

