import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('')
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                workers=8,
                # device='0,1',
                # resume='',
                project='',
                name='',
                )