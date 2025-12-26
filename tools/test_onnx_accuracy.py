#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import os
import struct
import sys

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 784)
    return images

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def test_onnx(model_path, images_path, labels_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    print(f"Loading MNIST data...")
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    
    # Preprocessing: Match C++ setting (normalize = false)
    print("Preprocessing images (scale only)...")
    images = images.astype(np.float32) / 255.0
    
    print(f"Running inference on {len(images)} samples...")
    correct = 0
    batch_size = 100
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Run inference
        outputs = session.run(None, {input_name: batch_imgs})
        logits = outputs[0]
        
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == batch_labels)
        
        if (i + batch_size) % 1000 == 0:
            print(f"Progress: {i + batch_size}/{len(images)}")

    accuracy = (correct / len(images)) * 100
    print(f"\n=== ONNX Test Result ===")
    print(f"Total Samples: {len(images)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}%")
    
    # Safety Check: If accuracy is 10%, it's random guessing
    if accuracy < 90:
        print("Warning: Accuracy is lower than expected! Check normalization or model export.")

if __name__ == "__main__":
    model = "exports/mnist_surrogate_best.onnx"
    if len(sys.argv) > 1:
        model = sys.argv[1]
        
    test_onnx(
        model, 
        "data/t10k-images-idx3-ubyte", 
        "data/t10k-labels-idx1-ubyte"
    )
