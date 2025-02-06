# Crea la cartella del progetto
mkdir drone_counter_project
cd drone_counter_project

# Crea l'ambiente virtuale
python -m venv venv

# Attiva l'ambiente (Windows)
venv\Scripts\activate

# Attiva l'ambiente (Linux/macOS)
source venv/bin/activate

# Installa le dipendenze
pip install opencv-python flask numpy ultralytics yt-dlp opencv-python-headless

# Salva le dipendenze in un file requirements.txt
pip freeze > requirements.txt

# Aggiorna il requirements.txt
echo "ffmpeg-python==0.2.0" >> requirements.txt

# Installa le nuove dipendenze
pip install ffmpeg-python

# Esegui l'applicazione dopo aver configurato
python drone_counter.py