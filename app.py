import cv2
from ultralytics import YOLOv10
import telnetlib
import time
import json

#prerequisites
# install brew
# install pyenv using brew
# install python 3.9.19 using pyenv
# install miniconda / anaconda 
# init miniconda / anaconda
# make sure "conda" is found in (zsh/bash)'s $PATH - create ~/.bashrc or ~/.zshrc and run "conda init" otherwise
# follow instructions on https://github.com/THU-MIG/yolov10 under heading "Installation"

enableVideoOutput = True

def setWindowSizeFromFeedFrameSize (cap, cv2):
    #get image size from frame
    ret, sizingImage = cap.read()
    height, width = sizingImage.shape[:2]
    print(f"Stream size: {width}x{height}")
    cv2.resizeWindow("RTSPStream", width, height)  # Width, Height
    if not ret:
        print("Failed to grab frame")
        exit

def main():
    with open('camera_credentials') as camCredsJson:
        camCredentials = json.load(camCredsJson)

    with open('zoneminder_credentials') as zoneminderCredsJson:
        zoneminderCredentials = json.load(zoneminderCredsJson)

    # RTSP URL
    rtsp_url = f"rtsp://{camCredentials['userId']}:{camCredentials['password']}@{camCredentials['host']}:{camCredentials['port']}/cam/realmonitor?channel={camCredentials['channel']}&subtype={camCredentials['subtype']}"
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    tn = telnetlib.Telnet(zoneminderCredentials['host'], zoneminderCredentials['port'])
 
    alarmStartMessage = "1|on|10|AI DETECT|insert classes here|\n"
    alarmStopMessage = "1|cancel|10|||\n"
    startTime = 0
    isStarted = False

    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
    else:
        print("RTSP stream opened successfully")
        if (enableVideoOutput):
            cv2.namedWindow("RTSPStream", cv2.WINDOW_NORMAL)
            setWindowSizeFromFeedFrameSize(cap, cv2)

        model = YOLOv10.from_pretrained(f'jameslahm/yolov10m')
        print(model.names)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                results = model.predict(source=frame, classes=[0,1,3,5], imgsz=(480,704), conf=0.7, save_conf=True)

                annotatedFrame = results[0].plot();
                now = int(time.time());
                
                for detection in results[0].boxes:
                    class_id = int(detection.cls)
                    class_name = model.names[class_id]
                    if (not isStarted):
                        print(f"Detected {class_name}. Triggering alarm at {now}")
                        tn.write(alarmStartMessage.encode('ascii'))
                        startTime = now
                        isStarted = True
                        print(f"TIME {startTime}")

                if (isStarted and startTime + 60 < now):
                    print(f"STOPPING AT {now}")
                    isStarted = False
                    tn.write(alarmStopMessage.encode('ascii'))

                # Display the frame
                if (enableVideoOutput):
                    cv2.imshow("RTSPStream", annotatedFrame)

                # Exit on pressing 'q'
                if enableVideoOutput and cv2.waitKey(1) & 0xFF == ord('q'):
                    cleanup(cv2, cap, tn, alarmStopMessage)
                    break
        except KeyboardInterrupt:
            cleanup(cv2, cap, tn, alarmStopMessage)
       

def cleanup(cv2, cap, tn, alarmStopMessage):
    # Release the capture and close windows
    print("EXITING")
    tn.write(alarmStopMessage.encode('ascii'))
    tn.close()
    cap.release()
    cv2.destroyAllWindows()

main()