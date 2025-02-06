import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from threading import Thread, Lock
import time
from ultralytics import YOLO
import argparse
import logging
import psutil
from pytube import YouTube
import pafy
import yt_dlp
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from datetime import datetime, timedelta
from collections import deque
import torch
import yaml

app = Flask(__name__)
lock = Lock()

# Variabili globali per il conteggio
total_count = 0
current_in_zone = 0
last_update = time.time()

# Modifica la logica di rilevamento dispositivo
def get_device():
    print("Verificando disponibilità CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        cuda_id = 0  # o specifica l'ID della GPU se ne hai multiple
        torch.cuda.set_device(cuda_id)
        print(f"Usando GPU: {torch.cuda.get_device_name(cuda_id)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return f'cuda:{cuda_id}'
    else:
        print("CUDA non disponibile, usando CPU")
        return 'cpu'

# Inizializzazione modello
device = get_device()
model = YOLO('yolov8n.pt', task='detect', verbose=False)
model.to(device)
half = device.startswith('cuda')

if device.startswith('cuda'):
    torch.cuda.empty_cache()  # Pulisci la memoria GPU

# Modifica la configurazione del tracker
TRACKER_CONFIG = {
    'tracker_type': 'bytetrack',
    'track_high_thresh': 0.6,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 15,
    'match_thresh': 0.7,
    'frame_rate': 30,
    'proximity_thresh': 0.2,
    'fuse_score': True
}

# Salva la configurazione in un file YAML
with open('bytetrack_custom.yaml', 'w') as f:
    yaml.dump(TRACKER_CONFIG, f)

# Configura il logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system.log')
    ]
)

class VideoProcessor:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.is_youtube = any(x in stream_url for x in ['youtube.com', 'youtu.be'])
        self.cap = None  # Inizializza come None
        self._init_capture()  # Metodo separato per l'inizializzazione
        
        self.detection_zone = []
        self.current_hour_max = 0
        self.current_hour_str = datetime.now().strftime("%H:00")
        self.hourly_counts = deque(maxlen=24)
        self._init_hourly_data()
        self.tracked_ids = {}
        self.cooldown = 5  # Secondi tra conteggi per lo stesso ID
        self.min_confidence = 0.6  # Soglia minima di confidenza
        self.last_positions = {}
        self.movement_threshold = 20  # Pixel minimi per considerare un movimento valido
        self.ENTRY_DIRECTION = 'top'  # 'top', 'bottom' o 'any'
        self.ENTRY_THRESHOLD = 0.25  # 25% del frame
        self.dead_ids_history = {}  # Tiene traccia degli ID rimossi e loro ultima posizione
        self.reid_threshold = 50    # Distanza massima in pixel per il re-identification
        self.max_id_seen = 0  # Aggiungi questa linea
        self.active_connections = 0
        self.stream_lock = Lock()

    def _init_capture(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)  # Aggiungi un piccolo delay per il rilascio
        
        self.cap = cv2.VideoCapture()
        
        if self.stream_url.isdigit():
            # Configurazione webcam
            self.cap.open(int(self.stream_url))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            # Configurazione stream RTSP/RTMP
            if not self.stream_url.startswith('rtsp://'):
                self.stream_url = f"rtsp://{self.stream_url}"
            
            # Costruisci correttamente i parametri di connessione
            base_url = self.stream_url.split('?')[0]
            params = {
                'rtsp_transport': 'tcp',  # Più affidabile di UDP
                'buffer_size': '1024',
                'timeout': '5000000',
                'analyzeduration': '1000000',
                'probesize': '1000000'
            }
            
            query = '&'.join([f"{k}={v}" for k, v in params.items()])
            self.stream_url = f"{base_url}?{query}"
            
            # Configurazione ottimizzata
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.open(self.stream_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logging.error(f"Connessione fallita a: {self.stream_url}")
            raise ValueError(f"Impossibile aprire il dispositivo: {self.stream_url}")
        else:
            logging.info(f"Connessione riuscita a: {self.stream_url}")
        
        self.detection_zone = []
        self.current_hour_max = 0
        self.current_hour_str = datetime.now().strftime("%H:00")
        self.hourly_counts = deque(maxlen=24)
        self._init_hourly_data()
        self.tracked_ids = {}
        self.cooldown = 5  # Secondi tra conteggi per lo stesso ID
        self.min_confidence = 0.6  # Soglia minima di confidenza
        self.last_positions = {}
        self.movement_threshold = 20  # Pixel minimi per considerare un movimento valido
        self.ENTRY_DIRECTION = 'top'  # 'top', 'bottom' o 'any'
        self.ENTRY_THRESHOLD = 0.25  # 25% del frame
        self.dead_ids_history = {}  # Tiene traccia degli ID rimossi e loro ultima posizione
        self.reid_threshold = 50    # Distanza massima in pixel per il re-identification
        self.max_id_seen = 0  # Aggiungi questa linea

    def _init_hourly_data(self):
        now = datetime.now()
        self.hourly_counts.clear()
        for i in range(24):
            self.hourly_counts.appendleft({
                'hour': (now - timedelta(hours=i)).strftime("%H:00"),
                'max_id': 0
            })

    def process_stream(self):
        with self.stream_lock:
            self.active_connections += 1
            try:
                if self.active_connections > 1:
                    logging.warning(f"Connessioni attive: {self.active_connections}")
                    return
                global total_count, current_in_zone, last_update
                tracked_objects = {}
                object_id = 0
                frame_count = 0
                start_time = time.time()

                current_frame_ids = set()
                current_confidences = {}

                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.warning("Frame non ricevuto")
                        self._init_capture()  # Reinizializza la connessione
                        ret, frame = self.cap.read()  # Ritenta una volta
                        if not ret:
                            logging.error("Impossibile ripristinare lo stream")
                            return

                    height, width = frame.shape[:2]
                    self.detection_zone = [(0, 0), (width, 0), (width, height), (0, height)]

                    results = model.track(
                        frame,
                        persist=True,
                        tracker='bytetrack_custom.yaml',
                        classes=[0],
                        conf=0.6,
                        iou=0.5,
                        imgsz=480,  # Ridotto per maggiore velocità
                        device=device,
                        half=half,
                        verbose=False
                    )
                    
                    current_objects = []
                    if results[0].boxes.id is not None:
                        for box, id, conf in zip(results[0].boxes, 
                                               results[0].boxes.id.cpu().numpy(),
                                               results[0].boxes.conf.cpu().numpy()):
                            if conf < self.min_confidence:
                                continue
                            
                            id = int(id)
                            current_frame_ids.add(id)
                            current_confidences[id] = conf
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            center_x = int((x1+x2)//2)
                            center_y = int((y1+y2)//2)
                            current_objects.append((center_x, center_y))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame, f"ID: {id}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                            # Calcola lo spostamento dall'ultima posizione
                            if id in self.last_positions:
                                prev_x, prev_y = self.last_positions[id]
                                distance = ((center_x - prev_x)**2 + (center_y - prev_y)**2)**0.5
                                if distance < self.movement_threshold:
                                    continue  # Ignora oggetti fermi

                            self.last_positions[id] = (center_x, center_y)

                            # Modifica nella logica di conteggio
                            if self.ENTRY_DIRECTION == 'top' and center_y > height * self.ENTRY_THRESHOLD:
                                continue
                            elif self.ENTRY_DIRECTION == 'bottom' and center_y < height * (1 - self.ENTRY_THRESHOLD):
                                continue

                    in_zone = 0
                    for (x, y) in current_objects:
                        if self.is_in_zone(x, y):
                            in_zone += 1

                    with lock:
                        # Sostituisci la logica di entered con il tracking degli ID
                        current_time = time.time()
                        new_entries = 0
                        
                        # Filtra solo gli ID nella zona di rilevamento
                        valid_ids = set()
                        for (x, y), id in zip(current_objects, current_frame_ids):
                            if self.is_in_zone(x, y):
                                valid_ids.add(id)
                        
                        # Aggiorna solo gli ID nella zona
                        for id in valid_ids:
                            # Controllo re-identification spaziale
                            is_reid = False
                            for dead_id, (dead_time, dead_pos) in self.dead_ids_history.items():
                                distance = ((center_x - dead_pos[0])**2 + (center_y - dead_pos[1])**2)**0.5
                                if distance < self.reid_threshold and (current_time - dead_time) < self.cooldown:
                                    is_reid = True
                                    self.tracked_ids[id] = dead_time  # Mantieni il timestamp originale
                                    break
                            
                            if not is_reid and (id not in self.tracked_ids or (current_time - self.tracked_ids[id]) > self.cooldown):
                                new_entries += 1
                                self.tracked_ids[id] = current_time
                                logging.info(f"Conteggio valido ID:{id}")
                            
                            # Sostituisci la logica di incremento con il tracking dell'ID massimo
                            if current_frame_ids:
                                current_max_id = max(current_frame_ids)
                                if current_max_id > self.max_id_seen:
                                    self.max_id_seen = current_max_id
                                total_count = self.max_id_seen
                                current_in_zone = len(valid_ids)
                                last_update = current_time

                                # Pulizia degli ID non più presenti
                                dead_ids = set(self.tracked_ids.keys()) - valid_ids
                                for id in dead_ids:
                                    del self.tracked_ids[id]
                                    if id in self.last_positions:
                                        del self.last_positions[id]

                                # Aggiorna la history degli ID rimossi
                                for id in dead_ids:
                                    if id in self.last_positions:
                                        self.dead_ids_history[id] = (current_time, self.last_positions[id])
                            
                                # Pulisci la history vecchia
                                self.dead_ids_history = {k:v for k,v in self.dead_ids_history.items() 
                                                        if (current_time - v[0]) < self.cooldown}

                    cv2.polylines(frame, [np.array(self.detection_zone)], True, (0,255,0), 2)
                    
                    # Modifica la sezione di aggiornamento del grafico
                    current_hour = datetime.now().strftime("%H:00")
                    
                    # Aggiorna il massimo dell'ora corrente
                    if valid_ids:
                        current_max = max(valid_ids)
                        if current_max > self.current_hour_max:
                            self.current_hour_max = current_max
                        
                    # Controllo cambio ora
                    if current_hour != self.current_hour_str:
                        # Trova e aggiorna l'entry nell'ora precedente
                        for entry in self.hourly_counts:
                            if entry['hour'] == self.current_hour_str:
                                entry['max_id'] = max(entry['max_id'], self.current_hour_max)
                                break
                        self.current_hour_str = current_hour
                        self.current_hour_max = 0

                    # Aggiorna in tempo reale l'entry corrente
                    for entry in self.hourly_counts:
                        if entry['hour'] == self.current_hour_str:
                            entry['max_id'] = max(entry['max_id'], self.current_hour_max)
                            break

                    ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            finally:
                self.active_connections -= 1
                if self.active_connections == 0:
                    self._init_capture()  # Rilascia la connessione solo se nessuno sta guardando

    def is_in_zone(self, x, y):
        return cv2.pointPolygonTest(np.array(self.detection_zone), (x,y), False) >= 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_processor.process_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    def generate():
        while True:
            with lock:
                yield f"data: {{\"total\": {total_count}, \"current\": {current_in_zone}}}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/history')
def history_data():
    return jsonify(list(video_processor.hourly_counts))

@app.route('/reset')
def reset_count():
    global total_count, current_in_zone
    with lock:
        total_count = 0
        current_in_zone = 0
        video_processor.max_id_seen = 0
        video_processor.current_hour_max = 0
        video_processor._init_hourly_data()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    # Aggiungi questo prima di iniziare
    print(f"Device utilizzato: {device}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU in uso: {torch.cuda.get_device_name(0)}")
        print(f"Half precision: {half}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default="0", 
                       help='Indice webcam (0,1..) o URL stream (rtsp://..., rtmp://...)')
    args = parser.parse_args()
    
    video_processor = VideoProcessor(args.url)
    try:
        app.run(host='0.0.0.0', port=5001, use_reloader=False)  # Disabilita auto-reload
    finally:
        video_processor._init_capture()  # Rilascia le risorse all'uscita 