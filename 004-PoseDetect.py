import cv2
from ultralytics import YOLO

def draw_points(image, points):
    h, w = image.shape[:2]  # Altura y anchura de la imagen
    for i,point in enumerate(points):
        x, y = point  # Coordenadas normalizadas
        x_pixel = int(x * w)
        y_pixel = int(y * h)
        cv2.circle(image, (x_pixel, y_pixel), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(image, str(i + 1), (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Inicializar YOLOv8 con el modelo pre-entrenado para la estimación de poses
model = YOLO('yolov8n-pose.pt')  # Asegúrate de que este sea el modelo correcto
data_izq=[]
data_der=[]
while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la estimación de poses con YOLO
    results = model(frame,device="cuda")
    print(results[0].keypoints.xyn.cpu().shape)

    # Dibujar los resultados en el frame
    #frame_with_results = results.render()[0]
    points=results[0].keypoints.xyn.cpu().numpy()[0]
    dif_derecha=points[7][1]-points[9][1]
    dif_izquierda=points[8][1]-points[10][1]
    data_izq.append(dif_izquierda)
    data_der.append(dif_derecha)

    draw_points(frame, points)

    # Mostrar el frame con los resultados
    cv2.imshow('YOLOv8 Pose Estimation', frame)

    # Romper el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
