from ultralytics import YOLO
import cv2
# cargar el modelo de yolo
model = YOLO('runs/detect/destorrnillador_pinzas_model/weights/best.pt')
'''
cargar el video ->   
    0 -> abra la camara principal 
    "nombre de video" -> el video
'''
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No camara")
    exit()

while True:
    # la camara cpturo ?  , la caputra
    ret , frame = cap.read()   
    
    if not ret:
        break
    # predecimos
    results = model(frame)
    # iteramos resultados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # obtener las cajas o boxes
            x1, y1, x2, y2  = box.xyxy[0]
            # convertirlo a entero
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # porcentaje de confianza de la deteccion
            confidence = float(box.conf[0])
            #clase a la que pertenece
            class_id = int(box.cls[0])  
            #extraer en nombre la claser
            class_name = model.names[class_id]
            # evaluamos porcentaje de confianza
            if confidence > 0.8:
                color = ( 0, 255 , 0 )
                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )
                text = f'{class_name} {confidence:.2f}'

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 -10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
