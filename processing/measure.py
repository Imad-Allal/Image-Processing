import numpy as np

def michelson(img):
    min_val = np.min(img)
    max_val = np.max(img)

    michelson_ms = (max_val - min_val) / (max_val + min_val)
    print(f'La mesure de Michelson : {michelson_ms}')

def rms(img):
    rms_ms = np.std(img) / np.mean(img)
    print(f'La mesure RMS : {rms_ms}')