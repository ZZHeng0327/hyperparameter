from ultralytics import YOLO
model=YOLO("C:\\Users\\Mikey\\runs\\detect\\train\\weights\\best.pt")
model.export(
    format="onnx",
    dynamic=False,
    batch=1,
    imgsz=[640,640],
    opset=11,
    nms=False,

)