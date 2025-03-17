from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'runs/train/exp20/weights/best.pt')
    model.predict(source=r'5cce9ca3e5ac6.jpg',
                  save=True,
                  show=True,
                  )
