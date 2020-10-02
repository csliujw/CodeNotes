# Demo

## 安裝

pip install opencv-python  

pip install opencv-contrib-python  识别物体用的

opencv官网下载安装 https://nchc.dl.sourceforge.net/project/opencvlibrary/4.4.0/opencv-4.4.0-vc14_vc15.exe

直接看视频写个Demo

检测图片中的人脸

```python
import cv2 as cv

CV_URL = "D:\Program Files\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"


class demo:
    def __init__(self, img_src):
        self.img = cv.imread(img_src)

    # 先要获取人脸特征 再进行检测
    # 提取出图像的细节【特征】，用两个图像对应特征的欧式距离来度量相似度
    def face_detection(self):
        gray = cv.cvtColor(self.img, code=cv.COLOR_BGR2GRAY)
        face_obj = cv.CascadeClassifier(CV_URL)
        faces = face_obj.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv.rectangle(self.img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv.imshow('result', self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    d = demo('gray_img.jpg')
    d.face_detection()
```

检测图片中的多张人脸

````python
import cv2 as cv

CV_URL = "D:\Program Files\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"


class demo:
    def __init__(self, img_src):
        self.img = cv.imread(img_src)

    # 先要获取人脸特征 再进行检测
    # 提取出图像的细节【特征】，用两个图像对应特征的欧式距离来度量相似度
    def face_detection(self):
        gray = cv.cvtColor(self.img, code=cv.COLOR_BGR2GRAY)
        face_obj = cv.CascadeClassifier(CV_URL)
        faces = face_obj.detectMultiScale(gray, scaleFactor=1.005, minNeighbors=3, maxSize=(88, 88), minSize=(24, 24))
        for x, y, w, h in faces:
            cv.rectangle(self.img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv.circle(self.img, center=(int(x + w / 2), int(y + h / 2)), radius=int(w / 2), color=(255, 0, 0))
            print(w, h)
        cv.imshow('result', self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    d = demo('mult_face.jpg')
    d.face_detection()
````

检测视频中的人脸

```python
import cv2 as cv

CV_URL = "D:\Program Files\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"


def face_detection(img):
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    face_obj = cv.CascadeClassifier(CV_URL)
    faces = face_obj.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv.imshow("result", img)


c_video = cv.VideoCapture("face.mp4")
while True:
    flag, frame = c_video.read()
    if not flag:
        break
    face_detection(frame)
    if ord('q') == cv.waitKey(10):
        break
cv.destroyAllWindows()
c_video.release()
```

