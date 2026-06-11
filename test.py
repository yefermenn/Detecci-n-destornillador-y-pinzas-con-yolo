from ultralytics import YOLO

model = YOLO('runs/detect/destorrnillador_pinzas_model/weights/best.pt')

metrics = model.val(data="data.yaml", split="test")

print("test finalizado")