from ultralytics.utils.ops import scale_boxes
import onnxruntime as ort
import numpy as np
import cv2
def letter_box (img,size):
    h, w = img.shape[:2]
    scale = min((float)(size/h), (float)(size/w))
    h = int(h * scale)
    w = int(w * scale)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    up_pad = (640 - h) // 2
    down_pad = 640 - h - up_pad
    left_pad = (640 - w) // 2
    right_pad = 640 - w - left_pad
    img = cv2.copyMakeBorder(img, up_pad, down_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=114)
    return img,scale,left_pad,up_pad

# =======================外部NMS============================
def NMS(img):
    img = np.squeeze(img, axis=0)
    img = np.transpose(img, (1, 0))
    cx, cy, w, h, class_score = img[:,0], img[:,1], img[:,2], img[:,3], img[:,4:]

    class_max = np.max(class_score, axis=1)
    class_idx = np.argmax(class_score, axis=1)
    mask = class_max > 0.25

    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    score = class_max[mask]
    class_idx = class_idx[mask]

    if len(class_idx) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    boxs_location = np.stack([x1, y1, x2, y2], axis=1)

    final_boxes = []
    final_scores = []
    final_class = []

    for c in np.unique(class_idx):
        c_mask = c == class_idx
        b = boxs_location[c_mask]
        s = score[c_mask]

        # 转成NMSBoxes要求的 [x1,y1,w,h]
        b_cv = np.zeros_like(b)
        b_cv[:,0] = b[:,0]
        b_cv[:,1] = b[:,1]
        b_cv[:,2] = b[:,2] - b[:,0]
        b_cv[:,3] = b[:,3] - b[:,1]

        # ==========核心修复：把numpy float转原生python float==========
        boxes_list = b_cv.tolist()
        scores_list = [float(x) for x in s.tolist()]

        keep_idx = cv2.dnn.NMSBoxes(boxes_list, scores_list, 0.25, 0.45)
        if len(keep_idx) == 0:
            continue

        keep_idx = keep_idx.flatten()
        final_boxes.extend(b[keep_idx].tolist())
        final_scores.extend(s[keep_idx].tolist())
        final_class.extend([c] * len(keep_idx))

    return np.array(final_boxes), np.array(final_scores), np.array(final_class)

# =================读入原图==================
img0=cv2.imread("C:\hand_gesture_dataset\images\\train\\0002.png")
img=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
# =========letter_box 记录数据===============
img,gain,pad_x,pad_y=letter_box(img,640)
# =============归一化处理====================
img = img.astype(np.float32) / 255.0
img=np.transpose(img,(2,0,1))
img=np.expand_dims(img,axis=0)

# =========onnxruntime会话推理===============
session=ort.InferenceSession(r"C:\Users\Mikey\runs\detect\train\weights\dynamic_int8.onnx", providers=["CPUExecutionProvider"])
input_name=session.get_inputs()[0].name
output_name=session.get_outputs()[0].name
[output0]=session.run([output_name],{input_name:img})
box_location,text1,text2=NMS(output0)

# ==============原图坐标========================（不知道为什么我scale_boxes用不了）
boxes_origin = box_location.copy()
boxes_origin[:, [0,2]] -= pad_x
boxes_origin[:, [1,3]] -= pad_y
boxes_origin = boxes_origin / gain
h0, w0 = img0.shape[:2]
boxes_origin[:, 0] = np.clip(boxes_origin[:, 0], 0, w0)
boxes_origin[:, 1] = np.clip(boxes_origin[:, 1], 0, h0)
boxes_origin[:, 2] = np.clip(boxes_origin[:, 2], 0, w0)
boxes_origin[:, 3] = np.clip(boxes_origin[:, 3], 0, h0)

# ======================画图========================
for i in range(len(boxes_origin)):
    x1,y1,x2,y2 = boxes_origin[i].astype(int)
    cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
    text = f"{text2[i]} {text1[i]:.2f}"
    cv2.putText(
        img0,
        text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
cv2.imshow("out", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
