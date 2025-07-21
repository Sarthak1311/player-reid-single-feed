from ultralytics import YOLO
model = YOLO("yolo11l.pt")
# Detection
results = model.predict(source = "/Users/sarthaktyagi/Downloads/Football-Analysis-System-Using-YOLO11-main/15sec_input_720p.mp4", save=True)


