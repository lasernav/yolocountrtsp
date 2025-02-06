# Drone Counter System

Sistema avanzato di conteggio droni basato su YOLOv8 e BYTETrack per il tracciamento e il conteggio di oggetti in tempo reale da stream video.

## ✨ Funzionalità principali
- 🎯 Rilevamento preciso con YOLOv8
- 📈 Tracciamento continuo tramite BYTETrack
- 🆔 Conteggio progressivo basato sugli ID
- 📊 Dashboard web con grafici in tempo reale
- 🕒 Storico orario dei conteggi
- 📹 Supporto multi-sorgente (webcam/RTSP/YouTube)
- 🔄 Gestione errori e riconnessione automatica

## 🖥️ Requisiti di sistema
- **Python 3.8+**
- **NVIDIA GPU** (consigliata) con driver CUDA 11.8+
- **4GB+ RAM** (8GB consigliati per HD)
- **2GB+ spazio disco**
- Connessione internet per download modelli

## Installazione

1. Clona il repository:
bash
git clone https://github.com/tuo-utente/drone-counter.git
cd drone-counter


3. Installa dipendenze:
1. Creazione ambiente virtuale:
myenv
2. Attivazione ambiente:
activate
3. Disattivazione ambiente:
deactivate
# Creazione ambiente virtuale:
python3 -m venv drone_env

Esempio completo per Windows:
# Windows (Prompt dei comandi)
:: Crea ambiente
python -m venv drone_env

:: Attiva
drone_env\Scripts\activate.bat

:: Installa pacchetti
pip install -r requirements.txt

:: Disattiva
deactivate
# Linux/MacOS
# Crea ambiente
python3 -m venv drone_env

# Attiva
source drone_env/bin/activate

# Installa pacchetti
pip install -r requirements.txt

# Disattiva
deactivate
pip install -r requirements.txt

## ⚡ Configurazione rapida

Crea `config.yaml` per personalizzare:
yaml
tracker:
track_high_thresh: 0.6 # Soglia alta per tracking
track_low_thresh: 0.1 # Soglia minima per tracking
new_track_thresh: 0.6 # Soglia per nuovi track
track_buffer: 30 # Frame di memoria per oggetti persi
match_thresh: 0.8 # Soglia matching tra frame
## Troubleshooting


## 🚀 Utilizzo

Avvia con webcam (ID 0):
bash
python drone_counter.py --url 0

Per stream RTSP
bash
python drone_counter.py --url "rtsp://username:password@ip:port/stream"
esempio con webcam : 
python drone_counter.py --url "rtsp://admin:PASSSS@10.0.0.14:554/Streaming/Channels/101"


:
bash
http://localhost:5001



**Comandi utili:**
- `--half`: Abilita precisione mista (performance)
- `--imgsz 640`: Dimensione elaborazione frame
- `--conf 0.5`: Soglia minima confidenza detection

## 🗂️ Struttura progetto

├── drone_counter.py # Core application
├── bytetrack_custom.yaml # Tracker configuration
├── templates/ # Web templates
│ └── index.html # Main interface
├── requirements.txt # Dependencies
└── README.md # This document
.

## 🛠️ Troubleshooting

| **Problema**              | **Soluzione**                                                                 |
|---------------------------|-------------------------------------------------------------------------------|
| Connessione video fallita | Verifica URL/porte • Disabilita firewall                                      |
| ID non incrementano       | Aumenta `track_buffer` • Modifica `new_track_thresh`                         |
| Alto uso CPU/GPU          | Riduci `--imgsz` • Usa `--half`                                              |
| Latenza elevata           | Disabilita visualizzazione frame (`--no-show`)                                |

## 📄 Licenza
Distribuito sotto licenza **[MIT](LICENSE)**.  
_Sviluppato da [Roberto Navoni] - 2025

> ⚠️ **Nota:** Richiede modello YOLOv8n.pt che viene scaricato automaticamente al primo avvio.