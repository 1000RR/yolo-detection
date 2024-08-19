import cv2
from ultralytics import YOLOv10


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
    # RTSP URL
    rtsp_url = "rtsp://admin:Hackathon$@hackathoncam.lan:554/cam/realmonitor?channel=1&subtype=1"
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
    else:
        print("RTSP stream opened successfully")
        cv2.namedWindow("RTSPStream", cv2.WINDOW_NORMAL)
        setWindowSizeFromFeedFrameSize(cap, cv2)

        model = YOLOv10.from_pretrained(f'jameslahm/yolov10m')
        print(model.names)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            results = model.predict(source=frame, classes=[0,1,2,3,5], imgsz=(480,704), conf=0.7, save_conf=True)

            annotatedFrame = results[0].plot();
            for result in results:
                for detection in result.boxes:
                    class_id = int(detection.cls)
                    class_name = model.names[class_id]
                    print(f"Detected {class_name}")


            # Display the frame
            cv2.imshow("RTSPStream", annotatedFrame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

main()