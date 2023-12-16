import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def draw_points(image, points):
    h, w = image.shape[:2]  # Altura y anchura de la imagen
    for i, point in enumerate(points):
        x, y = point  # Coordenadas normalizadas
        x_pixel = int(x * w)
        y_pixel = int(y * h)
        cv2.circle(image, (x_pixel, y_pixel), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(image, str(i + 1), (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)

def plot_to_image(fig):
    """ Convierte un gráfico de matplotlib en una imagen OpenCV """
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = canvas.buffer_rgba()
    X = np.asarray(buf, dtype=np.uint8)
    X = cv2.cvtColor(X, cv2.COLOR_RGBA2BGRA)

    return X

def init_plot():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=80)
    ax.set_ylim(-1, 1)  # Ajustar según sea necesario
    return fig, ax

def update_plot(ax, data_izq, data_der):
    ax.clear()
    ax.plot(data_izq, label='Izquierda', color='blue')
    ax.plot(data_der, label='Derecha', color='red')
    ax.legend(loc='upper right')
    return plot_to_image(ax.figure)

# Inicializar la cámara y el modelo YOLO
cap = cv2.VideoCapture(0)
model = YOLO('yolov8n-pose.pt')  # Asegúrate de que este sea el modelo correcto

# Inicializar datos para el gráfico y el gráfico en sí
data_izq, data_der = [], []
fig, ax = init_plot()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cuda")
    points = results[0].keypoints.xyn.cpu().numpy()[0]
    dif_derecha = points[7][1] - points[9][1]
    dif_izquierda = points[8][1] - points[10][1]
    data_izq.append(dif_izquierda)
    data_der.append(dif_derecha)
    if len(data_izq) > 50:
        data_izq.pop(0)
        data_der.pop(0)

    draw_points(frame, points)

    # Actualizar el gráfico a una tasa más baja para mejorar el rendimiento
    if len(data_izq) % 5 == 0:  # Actualizar el gráfico cada 5 frames
        graph_image = update_plot(ax, data_izq, data_der)
        frame[10:10+graph_image.shape[0], 10:10+graph_image.shape[1]] = graph_image

    cv2.imshow('YOLOv8 Pose Estimation con Gráfico', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
