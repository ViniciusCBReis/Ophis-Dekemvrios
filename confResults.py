import json
from collections import defaultdict

def load_predictions(file_path):
    with open(file_path, 'r') as file:
        predictions = json.load(file)
    return predictions

def organize_predictions(predictions):
    organized = defaultdict(list)
    for prediction in predictions:
        object_class = prediction['class']
        organized[object_class].append(prediction)
    return organized

def filter_predictions(predictions, confidence_threshold=0.5):
    filtered = [pred for pred in predictions if pred['confidence'] >= confidence_threshold]
    return filtered


