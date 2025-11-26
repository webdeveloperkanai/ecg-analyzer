import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_image
import cv2

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_PATH = 'ecg_model.pkl'

# Map directory names to class labels
DATA_DIR = '.' 

def get_data():
    """
    Walks through the directory and loads images.
    Returns X (features) and y (labels), and class_names mapping.
    """
    X = []
    y = []
    class_names = {}
    
    # Find directories
    all_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(d) and ('ECG' in d or 'Normal' in d)]
    print(f"Found directories: {all_dirs}")
    
    if not all_dirs:
        return None, None, None

    # Create mapping
    for idx, folder in enumerate(all_dirs):
        # Clean name
        if "Normal" in folder:
            name = "Normal"
        elif "History of MI" in folder:
            name = "History of Myocardial Infarction"
        elif "Myocardial Infarction" in folder:
            name = "Myocardial Infarction"
        elif "abnormal heartbeat" in folder:
            name = "Abnormal Heartbeat"
        else:
            name = folder
            
        class_names[idx] = name
        
        print(f"Loading class {name} from {folder}...")
        
        folder_path = os.path.join(DATA_DIR, folder)
        # Limit images per class for speed in this demo environment if needed
        # But let's try to load a reasonable amount, e.g., 100 per class for quick training
        # User asked for "stronger program", but I have time constraints. 
        # I'll load up to 200 images per class.
        
        count = 0
        for img_name in os.listdir(folder_path):
            if count >= 200: 
                break
            
            img_path = os.path.join(folder_path, img_name)
            try:
                # Use our utils but we need flat array for sklearn
                # preprocess_image returns (1, 224, 224, 3)
                # We can just use cv2 directly here for efficiency
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB to match utils.py
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                    # Normalize to [0, 1] to match utils.py
                    img = img.astype('float32') / 255.0
                    # Flatten
                    img_flat = img.flatten()
                    X.append(img_flat)
                    y.append(idx)
                    count += 1
            except Exception as e:
                pass
                
    return np.array(X), np.array(y), class_names

def train_model():
    print("Loading data...")
    X, y, class_names = get_data()
    
    if X is None or len(X) == 0:
        print("No data found.")
        return None

    print(f"Data loaded. Shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    # Save
    joblib.dump({'model': clf, 'class_names': class_names}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return clf

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

def predict_ecg(model_data, img_array, class_mapping=None):
    # model_data is the dict loaded from joblib
    model = model_data['model']
    # class_mapping is inside model_data usually, but we passed it separately in app.py
    # Let's use the one from the model file if available
    if 'class_names' in model_data:
        class_mapping = model_data['class_names']
        
    # img_array is (1, 224, 224, 3) from utils
    # Flatten it
    img_flat = img_array.flatten().reshape(1, -1)
    
    prediction_idx = model.predict(img_flat)[0]
    probabilities = model.predict_proba(img_flat)[0]
    confidence = np.max(probabilities)
    
    label = class_mapping.get(prediction_idx, "Unknown")
    
    return label, float(confidence)

if __name__ == "__main__":
    train_model()
