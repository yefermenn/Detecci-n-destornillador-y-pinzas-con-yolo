from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=2,
    name="destorrnillador_pinzas_model"
)

print("Entrenamiento finalizado")
print("Modelo guardado en: runs/detect/model/weights/best.pt")