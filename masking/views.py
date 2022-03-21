from django.http import HttpResponse
from django.shortcuts import render

from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading

xml = 'masking/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)

# 메인 페이지
def main(request):
    return render(request, 'main.html')

# 영상 송출 페이지
def home(request):
    return render(request, 'home.html')

@gzip.gzip_page
def video(request):
    try:
        # 응답 본문이 데이터를 계속 추가할 것이라고 브라우저에 알리고 브라우저에 원래 데이터를 데이터의 새 부분으로 교체하도록 요청
        # 즉, 서버에서 얻은 비디오가 jpeg 사진으로 변환되어 브라우저에 전달, 브라우저는 비디오 효과를 위해 이전 이미지를 새 이미지로 지속적 교체
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass


#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                              minNeighbors=5,
                                              minSize=(20, 20))
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.rectangle(image, (20, 20), (20 + 30, 20 + 30), (255, 0, 0), 3) # rectangle test

        frame_flip = cv2.flip(image, 1) # 좌우반전 flip
        _, jpeg = cv2.imencode('.jpg', frame_flip) # jpeg:인코딩 된 이미지

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()



def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mypage(request):
    return render(request, 'navBar/myPage/myPage.html')