

https://github.com/user-attachments/assets/87318b44-a20d-46a3-84e9-7394e6cd5746



# Vehicle Counting System

This project implements a vehicle counting system using the YOLOv8 object detection model and SORT tracker. The system processes video input to detect and track vehicles, counting the number that cross a designated line.

## Features

- **Real-time object detection**: Utilizes YOLOv8 for detecting vehicles in the video feed.
- **Object tracking**: Implements SORT (Simple Online and Realtime Tracking) to track detected vehicles across frames.
- **Vehicle counting**: Counts vehicles as they cross a specified line in the video.
- **Customizable**: Supports different input video streams and custom detection regions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/vehicle-counting-system.git
   cd vehicle-counting-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your video input**:
   Replace `cap = cv2.VideoCapture("1.mp4")` with `cap = cv2.VideoCapture(0)` to use a webcam, or specify your video file path.

2. **Set up your mask** (if needed):
   The mask (`mask2.png`) should match the region of interest in your video.

3. **Run the script**:
   ```bash
   python main.py
   ```

## Configuration

- **Line coordinates**: Modify the `line` variable to change the position of the counting line.
  ```python
  line = [500, 800, 950, 800]
  ```

- **Class names**: The `classNames` dictionary includes the detectable classes. Adjust the filtering logic as needed.

## Dependencies

- `numpy`
- `opencv-python`
- `cvzone`
- `ultralytics`
- `sort`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection model.
- [SORT](https://github.com/abewley/sort) for the tracking algorithm.
- [CVZone](https://github.com/cvzone/cvzone) for additional [OpenCV](https://github.com/opencv/opencv) utilities.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.
