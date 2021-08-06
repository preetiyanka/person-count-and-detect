import cv2

from Person_detector import PersonDetector

# Load Person Detector
pd =PersonDetector()

img=cv2.imread("C:/Users/preetiyanka/Desktop/New folder (4)/front-back (1).png")
pc=0
Person_boxes = pd.Person_vehicles(img)
person_count = len(Person_boxes)
pc+=person_count
print(pc)
for box in Person_boxes:
    x, y, w, h = box

    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
    cv2.putText(img, "Person : " + str(person_count), (20, 50), 0, 2, (100, 200, 0), 3)

       
cv2.imshow("Person", img)
cv2.waitKey(0)

