import os

os.chdir("yolov5_mod")
# os.system(r'python ..\.temp\test.py')
os.system(r'python detect.py --weights ..\models\facialRegion_yolov5s.pt --source ..\datasets\facialRegion\val\images\218.jpg --img 384 --line-thickness 1 --name exp --project ..\results\.temp\detect\facialRegion')
# os.system(r'python detect.py --weights ..\models\facialRegion_yolov5s.pt --source data\images\zidane.jpg --img 1280 --line-thickness 1 --name exp --project ..\results\detect\facialRegion\.temp')