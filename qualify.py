from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat
from glob import glob
import numpy as np
import cv2
path="D:\qualification"
img_folder=glob(f"{path}/*")

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self,img_folder):
        self.img_folder=img_folder
        self.index=0

    def letter_box(self,img, size):
        h, w = img.shape[:2]
        scale = min((float)(size / h), (float)(size / w))
        h = int(h * scale)
        w = int(w * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        up_pad = (640 - h) // 2
        down_pad = 640 - h - up_pad
        left_pad = (640 - w) // 2
        right_pad = 640 - w - left_pad
        img = cv2.copyMakeBorder(img, up_pad, down_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=114)
        return img, scale, left_pad , up_pad

    def get_next(self):
        if self.index>=len(self.img_folder):
            return None
        img=self.img_folder[self.index]
        img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img,_,_,_=self.letter_box(img,640)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img= np.expand_dims(img, axis=0)
        self.index+=1
        return {"images": img}

dr=MyCalibrationDataReader(img_folder)
quantize_static(
    model_input="C:\\Users\Mikey\\runs\detect\\train\weights\\best_sim.onnx",
    model_output="C:\\Users\Mikey\\runs\detect\\train\weights\int8_static.onnx",
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    nodes_to_exclude=["/model.23/Concat_2","/model.23/Mul_2","/model.23/Concat_3"]
     )