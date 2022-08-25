import cv2
import time

#our image 
img_file= '401_Gridlock.jpg'
video= cv2.VideoCapture('Pedestrians Compilation.mp4')
pTime=0
#our pretrained car and pedestrian classifer
car_tracker_file='car.xml'
pedestrian_tracker_file = 'person.xml'


#create car and pedestrian classifer
car_tracker =cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    # read the current frame
    (read_successful,frame) = video.read()

    # safe coding
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect car and pedestrians
    cars= car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)


    #draw rectangles around the cars
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x+1,y+2),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #print(cars)

    #draw rectangles around the pedestrians
    for(x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        #print(pedestrians)
    
    #frame rate
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0,),2)

    #Display image with face spotted
    cv2.imshow('ATS car detect',frame)
    
    #Don't autoclose
    key=cv2.waitKey(1)

    #stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the videoCapture object
video.release()















