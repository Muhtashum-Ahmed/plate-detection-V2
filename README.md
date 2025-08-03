# Smart Traffic Management System - Number Plate Detection 🚗📸

This project is part of a Final Year Project (FYP) aimed at building a Smart Traffic Management System that detects and tracks vehicles in real-time, identifies number plates, and logs overspeeding violations using computer vision techniques.

## 🚀 Features

- Real-time vehicle detection using **YOLOv8**
- Number plate recognition with **EasyOCR**
- Vehicle tracking via **SORT**
- Overspeeding detection using timestamp & position
- Logs violation data with time and plate number
- Cross-verification ready with university vehicle database

## 🧠 Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- EasyOCR
- SORT Tracker
- NumPy
- Pandas

## 🗂️ Project Structure

📁 your-project-root/
│
├── detect.py # Main script to detect & log number plates
├── best.pt # Trained YOLOv8 model
├── sort.py # SORT tracking module
├── requirements.txt # List of dependencies
├── .gitignore # Ignored files/folders


## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Create virtual environment (recommended):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate  # On Windows
```


3. **Install dependencies:**
  ```bash
  pip install ultralytics easyocr numpy opencv-python filterpy scikit-image matplotlib scipy


```
4. **Run the detection script:**
  ```bash
  python detect_v2.py

```
👨‍💻 Author
Muhtashum Ahmed – LinkedIn https://www.linkedin.com/in/muhtashum-ahmed-83882526b/
