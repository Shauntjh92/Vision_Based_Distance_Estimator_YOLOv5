import cv2

vidcap = cv2.VideoCapture("ORT.mp4")
success, image = vidcap.read()

while success:
    success,image = vidcap.read()

    #select coord
    tl=(500, 600)
    bl=(100, 1000)
    tr=(1300, 600)
    br=(1800, 1000)
    pts1 = [tl, bl, tr, br]
    cv2.circle(image,tl,5,(0,0,255),-1)
    cv2.circle(image,bl,5,(0,0,255),-1)
    cv2.circle(image,tr,5,(0,0,255),-1)
    cv2.circle(image,br,5,(0,0,255),-1)

    #geom transformation
    #pts1 =[tl,bl,tr,br]

    cv2.imshow("Frame",image)

    if cv2.waitKey(1)==27:
        break