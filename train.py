# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'datasets.yaml',
                imgsz=640,
                epochs=70,
                batch=64,
                workers=0,
                device='0',
                # optimizer='SGD',
                optimizer='AdamW',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                
#                 优化添加的
                hsv_h=0.015,  # 色相增强强度
                hsv_s=0.7,    # 饱和度增强
                hsv_v=0.4,    # 明度增强
                degrees=10.0,  # 旋转角度范围
                translate=0.2, # 平移幅度
                scale=0.9,     # 缩放幅度
                shear=2.0,     # 剪切幅度
                perspective=0.001,  # 透视变换
                flipud=0.5,    # 上下翻转概率
                fliplr=0.5,    # 左右翻转概率
                mosaic=1.0,    # Mosaic概率
                mixup=0.2,     # MixUp概率
                copy_paste=0.2,  # 复制粘贴增强
                auto_augment='rand-m9-mstd0.5-inc1',  # 自动增强策略
                erasing=0.1,   # 随机擦除概率
                crop_fraction=0.9,  # 裁剪比例
                )
