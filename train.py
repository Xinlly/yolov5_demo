import os

os.chdir("project")
print(os.getcwd())
os.system(r'python train.py --data ..\datasets\facialRegion\data.yaml --cfg ..\models\facialRegion_yolov5s.yaml --weights ..\models\yolov5s.pt --workers 4 --batch-size 4 --img 384 --name exp --project ..\results\train\.temp\facialRegion')