import cv2
import numpy as np
import time

# --------- USER CONFIG ----------
ONNX_MODEL = "yolov5s.onnx"   # put yolov5s.onnx here
COCO_NAMES = "coco.names"     # coco class names
CONF_THRESH = 0.4
NMS_THRESH = 0.45
CAMERA_ID = 0                 # change if needed
# --------------------------------

# Load class names
with open(COCO_NAMES, "r") as f:
    classes = [c.strip() for c in f.readlines()]

# Find index for 'cell phone' in coco (if present)
phone_class_indices = [i for i, name in enumerate(classes) if "phone" in name or "cell phone" in name or "cellphone" in name]
# fallback: try exact 'cell phone'
if not phone_class_indices:
    try:
        phone_class_indices = [classes.index("cell phone")]
    except:
        phone_class_indices = []

# Load ONNX model using OpenCV DNN
net = cv2.dnn.readNetFromONNX(ONNX_MODEL)
# Use CUDA if compiled OpenCV supports it (optional)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

def yolov5_infer(frame, net, conf_thres=CONF_THRESH, nms_thres=NMS_THRESH):
    """
    Returns list of detections: [ (x1,y1,x2,y2, conf, class_id), ... ]
    """
    img = frame.copy()
    h0, w0 = img.shape[:2]
    # Preprocess (same as YOLOv5 ONNX expected)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = 640 / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = 640 - new_unpad[0]
    dh = 640 - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img_padded = img_padded.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_padded, (2,0,1))
    blob = np.expand_dims(img_transposed, 0)

    net.setInput(blob)
    preds = net.forward()   # shape: (1, N, 85) or (1,25200,85) depending
    preds = np.squeeze(preds)

    if preds.ndim == 2 and preds.shape[1] == 85:
        boxes = []
        for det in preds:
            scores = det[4] * det[5:]  # objectness * class scores
            class_id = np.argmax(scores)
            conf = float(scores[class_id])
            if conf > conf_thres:
                cx, cy, w, h = det[0:4]
                # convert from relative? YOLOv5 ONNX outputs are normalized to 640 grid already.
                # The outputs are in original 640x640 scale (center x,y,w,h)
                x1 = float(cx - w/2)
                y1 = float(cy - h/2)
                x2 = float(cx + w/2)
                y2 = float(cy + h/2)
                boxes.append([x1, y1, x2, y2, conf, class_id])
        if not boxes:
            return []
        boxes = np.array(boxes)
        # Convert coordinates back to original image scale
        # Remove padding and scale:
        boxes_scaled = []
        for x1,y1,x2,y2,conf,class_id in boxes:
            x1 = (x1 - left) / r
            y1 = (y1 - top) / r
            x2 = (x2 - left) / r
            y2 = (y2 - top) / r
            # clamp
            x1 = max(0, min(x1, w0-1))
            y1 = max(0, min(y1, h0-1))
            x2 = max(0, min(x2, w0-1))
            y2 = max(0, min(y2, h0-1))
            boxes_scaled.append([int(x1), int(y1), int(x2), int(y2), conf, int(class_id)])
        # NMS
        boxes_np = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_scaled]).astype(np.float32)
        confidences = boxes[:,4].astype(np.float32)
        indices = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences.tolist(), conf_thres, nms_thres)
        final = []
        if len(indices) > 0:
            for i in indices.flatten():
                final.append(tuple(boxes_scaled[i]))
        return final
    else:
        return []

def detect_pen_by_shape(frame):
    """
    Heuristic pen detector:
    - Convert to grayscale, blur, Canny
    - Find contours, filter by aspect ratio (long thin) and area & solidity
    - Also check Hough lines to confirm linearity
    Returns list of bounding boxes for pens.
    """
    out_boxes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Dilate to connect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:   # ignore tiny
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        aspect = ww / float(hh) if hh>0 else 0
        rect_area = ww * hh
        solidity = area / float(rect_area) if rect_area>0 else 0

        # Heuristic thresholds (tweakable)
        if ( (aspect > 3.0 or aspect < 0.33) and area > 500 and solidity > 0.2):
            # To be more confident, check Hough lines inside this ROI
            roi = edges[y:y+hh, x:x+ww]
            if roi.size == 0:
                continue
            lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=30, minLineLength=int(max(ww,hh)*0.6), maxLineGap=10)
            if lines is not None:
                out_boxes.append((x,y,x+ww,y+hh))
            else:
                # allow some without lines (in case of solid pen)
                out_boxes.append((x,y,x+ww,y+hh))
    # Optionally merge overlapping boxes
    return out_boxes

# Start webcam
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Cannot open camera. Check CAMERA_ID or webcam.")
    exit()

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize display for speed if needed (comment out if you want full res)
    # frame = cv2.resize(frame, (960, 540))

    detections = yolov5_infer(frame, net)

    phone_count = 0
    # draw YOLO detections
    for (x1,y1,x2,y2,conf,class_id) in detections:
        # Map class_id to classes list length: YOLO preds class_id corresponds to index in classes
        cls_id = int(class_id)
        label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        if "phone" in label or "cell" in label:
            phone_count += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

    # Pen detection heuristic
    pen_boxes = detect_pen_by_shape(frame)
    pen_count = len(pen_boxes)
    for (x1,y1,x2,y2) in pen_boxes:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (200,0,0), 2)
        cv2.putText(frame, f"pen", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 1)

    # Show counts on top-left
    info = f"Phones: {phone_count}   Pens: {pen_count}"
    cv2.rectangle(frame, (5,5), (320,30), (0,0,0), -1)
    cv2.putText(frame, info, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # FPS
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if cur_time != prev_time else 0.0
    prev_time = cur_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Phone & Pen Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
