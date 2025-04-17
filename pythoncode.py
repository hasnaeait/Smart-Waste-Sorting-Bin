import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import serial  # Pour communiquer avec Arduino
import time
from ultralytics import YOLO

# === Connexion série avec Arduino (à adapter selon ton port COM) ===
arduino = serial.Serial('COM11', 9600, timeout=1)
time.sleep(2)  # Laisser le temps à Arduino de redémarrer

# === Charger le modèle YOLO ===
model = YOLO(r'C:\Users\hassnae\Downloads\python\runs\detect\modele_personnalise_yolov83\weights\best.pt', task='detect')

# === Fonction d'envoi d'e-mail ===
def send_email(bin_status):
    sender_email = "your_email@example.com"
    receiver_email = "person_in_charge@example.com"
    password = "your_password"

    subject = "Smart Bin Alert"
    body = f"The bin is currently {bin_status}. Please empty it soon."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# === Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error.")
    exit()

# === Boucle principale ===
threshold = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.predict(source=frame, conf=threshold, show=True)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                print(f"Detected: {label}")
                if label == "plastic":
                    send_email("FULL")
                elif label == "cardboard":
                    arduino.write(b'C')  # Commande pour le servo du pin 8
                    print("Cardboard detected. Command sent to Arduino.")
                elif label == "glass":
                    arduino.write(b'G')  # Commande pour le servo du pin 9
                    print("Glass detected. Command sent to Arduino.")
        else:
            print("No objects detected.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
