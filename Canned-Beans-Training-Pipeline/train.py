# import packages
import os
import shutil
from ultralytics import YOLO

# delete job_logs directory if exists
if os.path.exists("./job_logs"):
    shutil.rmtree("./job_logs")

# delete runs directory if exists
if os.path.exists("./runs"):
    shutil.rmtree("./runs")

# create job_logs directory
os.makedirs('./job_logs', exist_ok=True)

# initialize model
model = YOLO('yolov8n.pt')

# train model
model.train(
    data='./dataset/data.yaml',  
    epochs=50,                 
    imgsz=640,                
    batch=16,                   
    device=1,                  
    name='yolov8n-custom',
    patience=5,               
    workers=1,                 
    verbose=True,
    val=True,                
)