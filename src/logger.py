import logging
import os
from datetime import datetime

# Nur den Logs-Ordner anlegen
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Logdatei-Name
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"

# Vollständiger Dateipfad
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=LOG_FILE_PATH
)