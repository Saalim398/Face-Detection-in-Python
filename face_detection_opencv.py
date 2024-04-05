#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import matplotlib.pyplot as plt


# In[6]:


image = cv2.imread('E:users\\downloads\\PRJ Face Detection\\Data Set\\people1.jpg')


# In[7]:


display(image.shape)


# In[8]:


plt.imshow(image)
plt.show()


# In[9]:


print(image)


# In[10]:


cv2.imshow('', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[11]:


image = cv2.resize(image, (800, 600))


print (image.shape)


# In[12]:


image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('',image_gray)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[ ]:





# In[13]:


face_detector = cv2.CascadeClassifier('E:users\\downloads\\PRJ Face Detection\\Cascades\\haarcascade_frontalface_default.xml')
display (face_detector)


# In[14]:


detections = face_detector.detectMultiScale(image_gray)
display(detections)


# In[15]:


display (len(detections))


# In[16]:


for (x,y,w,h) in detections:
    cv2.rectangle(image_gray,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('', image_gray)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[17]:


for (x,y,w,h) in detections:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('', image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[19]:


detections = face_detector.detectMultiScale(image_gray,scaleFactor=1.09)
display(detections)
for (x,y,w,h) in detections:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('', image)
cv2.waitKey(10000)
plt.show()


# In[22]:


detections = face_detector.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=9)
display(detections)
for (x,y,w,h) in detections:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('', image)
cv2.waitKey(10000)
plt.show()


# In[24]:


detections = face_detector.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=9,minSize=(20,20), maxSize=(100,100))
display(detections)
for (x,y,w,h) in detections:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('', image)
cv2.waitKey(10000)
plt.show()


# In[25]:


#EYE DETECTION


# In[26]:


eye_detector= cv2.CascadeClassifier('E:users\\downloads\\PRJ Face Detection\\Cascades\\haarcascade_eye.xml')
display (eye_detector)


# In[27]:


image = cv2.resize(image, (1600,1000)) # Resize image
print(image.shape)
face_detections = face_detector.detectMultiScale(image, scaleFactor = 1.3, minSize = (30,30))
for (x, y, w, h) in face_detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

eye_detections = eye_detector.detectMultiScale(image, scaleFactor = 1.1, minNeighbors=10, maxSize=(60,60))

for (x, y, w, h) in eye_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[29]:


#car Detector


# In[31]:


car_detector = cv2.CascadeClassifier('E:users\\downloads\\PRJ Face Detection\\Cascades\\cars.xml')

display (car_detector)


# In[32]:


image = cv2.imread('E:users\\downloads\\PRJ Face Detection\\Data Set\\car.jpg')
display (image.shape)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[42]:


import cv2
image = cv2.imread('E:users\\downloads\\PRJ Face Detection\\Data Set\\car.jpg')
display(image.shape)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()

#Car Detector  

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = car_detector.detectMultiScale(image_gray, scaleFactor = 1.1, minNeighbors=8)                                      
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[45]:


clock_detector = cv2.CascadeClassifier('E:users\\downloads\\PRJ Face Detection\\Cascades\\clocks.xml')
display (clock_detector)

# Load image 

image = cv2.imread('E:users\\downloads\\PRJ Face Detection\\Data Set\\clock.jpg')
display (image.shape)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()

# Clock Detector 

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = clock_detector.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors=1)                                      
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[46]:


full_detector  = cv2.CascadeClassifier('E:users\\downloads\\PRJ Face Detection\\Cascades\\fullbody.xml')
display (full_detector)

# Load image 
image = cv2.imread('E:users\\downloads\\PRJ Face Detection\\Data Set\\people3.jpg')
display (image.shape)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()

# Body Detector 

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = full_detector.detectMultiScale(image_gray, scaleFactor = 1.05, minNeighbors=5,    minSize = (50,50)) 
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow('',image)
cv2.waitKey(10000)
plt.show()
cv2.destroyAllWindows()


# In[ ]:




