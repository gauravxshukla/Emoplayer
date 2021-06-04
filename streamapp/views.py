from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import  EmoDetect
# Create your views here.
emotion= []
gender= []
age= []

def index(request):
    return render(request, 'index.html')


def gen(camera):
    while True:
        frame2 = camera.get_frame()

        frame=frame2[0]
        #print("in views:")
        #print(frame2[1])
        if(len(frame2[1])==3):
            emotion.append(frame2[1][0])
            gender.append(frame2[1][1])
            age.append(frame2[1][2])

        
        
        
        #print("in views emo:")
        #print(emotion)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




def Emo_feed(request):
    return StreamingHttpResponse(gen(EmoDetect()),
                    content_type='multipart/x-mixed-replace; boundary=frame')
                    

def rend(request):
        #emotion=request.GET["Emotion"]
        #gender=request.GET["Gender"]
        #print("....")
        #print(emotion[-1])
        #print("....")
        if(emotion[-1]=='Happy' ):
            return render(request,"result.html")
        
        
        else:
            return render(request,"result2.html")

        
        