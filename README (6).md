# MagicCloakCV – Harry Potter Inspired Invisibility Cloak

MagicCloakCV is a real-time computer vision application built with **Python** and **OpenCV** that recreates the "invisibility cloak" effect from the Harry Potter films. By detecting a specific colored cloth (blue, by default) in the webcam feed and replacing it with a pre-captured background frame, the person wearing the cloth appears to vanish — just like Harry's invisibility cloak.

The project also includes an experimental **object detection module** (`project.py`) that uses a YOLOv5 ONNX model together with a classical shape-based heuristic to detect phones and pens in the webcam feed.

---

## Demo Preview

The application opens three windows:
1. **bars** – trackbars to live-tune the HSV color range used to detect the cloak
2. **Mask** – the binary mask showing the detected cloak region
3. **Harry's Cloak** – the final output feed with the cloak effect applied

---

## Features

- Real-time invisibility effect using HSV color segmentation
- Adjustable HSV trackbars (`lower_hue`, `upper_hue`, `lower_saturation`, `upper_saturation`, `lower_value`, `upper_value`) for tuning detection to any cloth color/lighting condition
- Noise removal using median blur and morphological opening/closing
- Largest-contour filtering to remove background color speckles
- On-the-fly background re-capture (press **`b`**) without restarting the program
- Clean exit with **`q`**
- Bonus module: YOLOv5-based phone detection + heuristic-based pen detection with live FPS counter

---

## How It Works (Invisibility Cloak — `projecto.py`)

1. **Background Capture** – When the program starts, it captures ~30 frames of the empty scene (without the person/cloth) to use as the static background.
2. **HSV Conversion** – Each incoming frame is converted from BGR to HSV color space, which separates color (Hue) from brightness (Value), making color-based segmentation more robust to lighting changes.
3. **Color Masking** – `cv2.inRange()` creates a binary mask that marks pixels within the selected HSV range (i.e., the cloak) as white and everything else as black.
4. **Noise Reduction** – Median blur and morphological operations (open, then close) clean up small false-positive/false-negative pixels in the mask.
5. **Largest Contour Extraction** – Only the single largest connected white region is kept, which removes any small blue-ish speckles elsewhere in the frame.
6. **Frame Compositing** – The current frame is masked to hide the cloak area, and the pre-captured background is masked to reveal only the cloak area. The two are blended together (`cv2.addWeighted`) to produce the final "invisible" output.

---

## How It Works (Bonus — `project.py`)

- **YOLOv5 ONNX Inference**: Loads a `yolov5s.onnx` model via OpenCV's DNN module, preprocesses each frame (letterbox resize to 640×640, normalize, transpose to CHW), runs a forward pass, and applies confidence thresholding + Non-Maximum Suppression to get final bounding boxes. Detections labeled "phone"/"cell" are counted and drawn.
- **Pen Detection Heuristic**: Uses Canny edge detection and contour analysis, filtering contours by aspect ratio (long & thin) and solidity, then confirms linearity with a probabilistic Hough Line Transform.
- Displays live phone/pen counts and FPS on screen.

---

## Requirements

- Python 3.7+
- A working webcam
- Packages:
  ```
  opencv-python
  numpy
  ```
- For the bonus detector (`project.py`) only:
  - `yolov5s.onnx` model file
  - `coco.names` class-names file

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## Project Structure

```
MagicCloakCV/
│
├── projecto.py        # Main invisibility cloak application
├── project.py          # Bonus: phone & pen detector (YOLOv5 + heuristics)
├── yolov5s.onnx         # (required only for project.py) YOLOv5 ONNX model
├── coco.names           # (required only for project.py) COCO class labels
└── README.md
```

---

## Usage

### 1. Run the Invisibility Cloak
```bash
python projecto.py
```
1. When prompted, step out of the camera's view so the background can be captured.
2. Step back in wearing a **blue** cloth/garment (default tuned range).
3. Use the trackbars in the **bars** window to fine-tune detection if the cloak effect looks patchy.
4. Press **`b`** at any time to re-capture the background (useful if lighting changes).
5. Press **`q`** to quit.

### 2. Run the Object Detector (Bonus)
```bash
python project.py
```
Place `yolov5s.onnx` and `coco.names` in the same directory before running. Press **`q`** or **`Esc`** to quit.

---

## Tuning Tips

| Trackbar | Purpose | Tip |
|---|---|---|
| `lower_hue` / `upper_hue` | Color range | Narrow this window to isolate the exact shade of your cloth |
| `lower_saturation` / `upper_saturation` | Colorfulness | Raise the lower bound to ignore washed-out/gray backgrounds |
| `lower_value` / `upper_value` | Brightness | Adjust for room lighting — darker rooms may need a lower `lower_value` |

For best results, use a **plain, evenly lit background** and a cloth color that doesn't appear elsewhere in the scene (skin tone, walls, clothing).

---

## Limitations

- Sensitive to lighting changes — recapturing the background (`b`) is often needed.
- Only tracks a single largest cloak region at a time (multiple cloaked people won't work well).
- The bonus detector requires an external YOLOv5 ONNX model that isn't bundled in this repository.

---

## Future Improvements

- Automatic HSV range detection by sampling a click on the cloth
- Support for multiple simultaneous cloak colors/regions
- Edge feathering for smoother cloak boundaries
- Replace classical pen heuristic with a trained lightweight detector

---

## Credits

Built using **OpenCV**, **NumPy**, and (for the bonus module) a **YOLOv5** ONNX model, inspired by the invisibility cloak from the *Harry Potter* series.
