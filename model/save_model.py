import pickle
from utils.logger import setup_logging

logging = setup_logging()

def save_model(model, filename="real_estate_model.pickle"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
        logging.info(f"Model saved as {filename}")