from LiveWebcam import LiveWebcamMain
from ProcessingRealTime import ProcessingRealTimeMain


whichFile = input("Would you like to use your webcam(1) or a preshot video(2)? ")
if(whichFile == '1'):
    LiveWebcamMain()
else:
    ProcessingRealTimeMain()

