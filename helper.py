from ultralytics import YOLO 
model = YOLO('/Users/sarthaktyagi/Downloads/Football-Analysis-System-Using-YOLO11-main/model/best.pt')
print(model.names)