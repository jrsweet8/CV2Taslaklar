#baslangic
import cv2
import numpy as np
# print("package ipmortaed")

####reading images webcam and videos
# imgtest = cv2.imread('C:/Users/jrswe/Desktop/opencv/resources/girl.jpg')
##resmi gostermek
# cv2.imshow('girl',imgtest)
# cv2.waitKey(0)

#videocyu bulmak

# cap = cv2.VideoCapture('C:/Users/jrswe/Desktop/opencv/resources/video.mp4')
##videoyu izlemek
# while True:
#     success, img = cap.read()
#     cv2.imshow('Video',img)
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break

##vebcam gosterme

# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)

# while True:
#     success, img = cap.read()
#     cv2.imshow('Video',img)
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break

##webcam izlemek







###Basic Functions


# img = cv2.imread('C:/Users/jrswe/Desktop/opencv/resources/girl.jpg')
# kernel = np.ones((5,5),np.uint8)

# #bri ve blury yapanlar
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBLur = cv2.GaussianBlur(imgGray,(7,7),0)

# #koseleri bulmak
# imgcanny = cv2.Canny(img,150,200)
# imgDilation = cv2.dilate(imgcanny,kernel,iterations =1)
# imgEroded = cv2.erode(imgDilation,kernel,iterations=1)


# cv2.imshow('gray image',imgGray)
# cv2.imshow('Blur image',imgBLur)
# cv2.imshow('canny image',imgcanny)
# cv2.imshow('Dilation image',imgDilation)
# cv2.imshow('Eroded image',imgEroded)
# cv2.waitKey(0)






###OpenCV Convention
# #boyutlari degsitirmek
# img = cv2.imread('C:/Users/jrswe/Desktop/opencv/resources/geralt.jpg')
# print(img.shape)

# imgResize = cv2.resize(img,(600,300))
# print(imgResize.shape)

# #kesmek
# imgcropped = img[0:200,200:500]

# cv2.imshow('image',img)
# #cv2.imshow('imageresiz',imgResize)
# cv2.imshow('image cropped',imgcropped)
# cv2.waitKey(0)





###Lines and Spahes also Text
# #siyah bir eksran
# img = np.zeros((512,512,3),np.uint8)
# #print(img.shape)
# #mavi bir ekran
# #img[:]= 255,0,0
# #kismi mavi bir ekran
# #img[200:300,100:300] = 255,0,0

# #biseyler cizmek
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
# cv2.circle(img,(400,50),30,(255,255,0),5)

# #biseyler yazmak
# cv2.putText(img," OPENCV ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),2)


# cv2.imshow('img',img)
# cv2.waitKey(0)






###WarpPerspective?
# #Bakis acini degistir!
# img = cv2. imread('C:/Users/jrswe/Desktop/opencv/resources/cards.jpg')

# #uretilen resmin boyutu
# width,height = 250,350
# #alinacak noktalar
# pts1 = np.float32([[78,27],[157,39],[40,125],[136,139]])
# #olusacak resim
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgoutput = cv2.warpPerspective(img,matrix,(width,height))

# cv2.imshow('cards',img)
# cv2.imshow('output',imgoutput)

# cv2.waitKey(0)



####Joining Images
# #bunu ekranda bircok resim gormek isterken kullanabilirsin
# #olasi problemlersu sekilde , burada resimlerin boyutlari ayarlanamaz o yuzden kocaman seyler olabilir
# img = cv2.imread('C:/Users/jrswe/Desktop/opencv/resources/girl.jpg')

# ###buradaki problemlerin cozumu icin izledigim videoda uzun bir fonksyon vardi
# ###eger ihtiyacin olursa giip o fonksyonu al....


# # imghor = np.hstack((img,img))
# # imgver = np.vstack((img,img))

# # cv2.imshow('Horrizontal',imghor)
# # cv2.imshow('Vertical',imgver)

# cv2.waitKey(0)







####ColorDetection

# #empyt fonk. tanimlanmasi
# def empty():
#     pass

# #Trackbarin yapilmasi ve cagrilmasi
# #Trackbar ile bu ozellikler gozlemlenir ve ardindan istenilen renk araligi tespit edilir.
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",99,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",123,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",112,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",33,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",212,255,empty)
# ##Trackbar while loopuna sokularak resim uzerinde yapilan oynamalarin yansitilmasi ve dolayisiyla aranilan degerin purussuz oldugu degerlere
# ##ulasmak hedeflenir burada ulasildiginda degerler kaydedilir.
# while True:
#     path = 'C:/Users/jrswe/Desktop/opencv/resources/hyundai.jpg'
#     img = cv2.imread(path)
#     imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min","TrackBars")  
#     v_max = cv2.getTrackbarPos("Val Max","TrackBars")  
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV,lower,upper)

#     #maskeden faydalanarak yeni resim olusturmak
#     imgResult = cv2.bitwise_and(img,img,mask=mask)

#     #bnularin hepsinden sonra bu resimler tek tek degl ama bir arada gosterilebilir daha onceki bolumerde beraber gosteren fonksyondan bahsetmistim...
#     cv2.imshow('img',img)
#     cv2.imshow('HSV',imgHSV)
#     cv2.imshow('Mask',mask)
#     cv2.imshow('imgResult',imgResult)
#     cv2.waitKey(1)









####Contour and Spahe Detection

###gerekli fonksyonun tanimlanmasi
# def getcontours(img):
#     #cv2 altinda tanimli olan findcountours fonksoynunun aldigi retrexternal parametresi 
#     #cok fazla olan dis koseleri falan buluyor bu dis koseler icin iyidir baskalari da var takbiki
#     #cahin approx ise hepsini al gibi bisey
#     contours,hierarchy =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         #sekil alaninin hesaplanmasi
#         area = cv2.contourArea(cnt)
#         #print(area)
#         if area >300:
#                     #sekillerin cevresien kendilerini cizdirip dogru mu yanlis mi gormek
#                     cv2.drawContours(imgcontour,cnt,-1,(255,0,0),3)
#                     peri = cv2.arcLength(cnt,True)
#                     #print(peri)
#                     approx = cv2.approxPolyDP(cnt,0.02*peri,True)
#                     print((approx))
#                     objCor = len(approx)
#                     #kutuyu cizmek icin on isleme
#                     x, y, w, h = cv2.boundingRect(approx)
                    
#                     if objCor ==3: objectType = "Tri"
#                     elif objCor ==4:
#                         aspRatio = w/float(h)
#                         if aspRatio > 0.9 and aspRatio <1.15: objectType ="Square"
#                         else: objectType ="Rectangle"
#                     elif objCor >4: objectType = "Circle"
#                     else:objectType = "None"
                    
#                     #kutunun cizilmesi
#                     cv2.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),2)
#                     #cizimlerin ne oldgunu soylemek
#                     cv2.putText(imgcontour,objectType,
#                     (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)


# path = "C:/Users/jrswe/Desktop/opencv/resources/shapes.png"
# img = cv2.imread(path)
# img = cv2.resize(img,(300,300))
# imgcontour = img.copy()
# #preprocessing ve graysacale cevirimi
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
# #canny edge detecdor kullanimi
# #blurry yaparak sekilleri sadece sekil olacak sekilde var etmeye calisiyoruz.
# imgcanny = cv2.Canny(imgBlur,179,179)
# getcontours(imgcanny)

# # cv2.imshow('Canny',imgcanny)
# # cv2.imshow('gray',imgGray)
# # cv2.imshow('Blur',imgBlur)
# # cv2.imshow('Original',img)
# cv2.imshow('contour',imgcontour)
# cv2.waitKey(0)










####Face Detection

# #farkli metodlarla insan yuzu buklunabilir bizim kullanacagimiz 
# #zaten cv2nun icinde olanlardan birtansei 
# #bu cascadeyi internetten bulup indridim ki kullanayim...
# faceCascade =cv2.CascadeClassifier("resources/haarcascade_frontalcatface.xml")


# img = cv2.imread('resources/face2.jpg')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(imgGray,1.1,4)
# #yuzun etrafina kutu yapmak

# for (x,y,w,h) in faces :
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)




# cv2.imshow('Result',img)
# cv2.waitKey(0)