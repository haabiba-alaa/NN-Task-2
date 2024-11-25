import numpy as np
import pandas as pd

def TTS(X, y, train_size_per_class=30, test_size_per_class=20, num_classes=3):
    y = np.array(y)  
    
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size), y] = 1
    
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    class_2_indices = np.where(y == 2)[0]
    
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    np.random.shuffle(class_2_indices)
    
    train_indices_0 = class_0_indices[:train_size_per_class]
    test_indices_0 = class_0_indices[train_size_per_class:train_size_per_class + test_size_per_class]
    
    train_indices_1 = class_1_indices[:train_size_per_class]
    test_indices_1 = class_1_indices[train_size_per_class:train_size_per_class + test_size_per_class]
    
    train_indices_2 = class_2_indices[:train_size_per_class]
    test_indices_2 = class_2_indices[train_size_per_class:train_size_per_class + test_size_per_class]
    
    train_indices = np.concatenate([train_indices_0, train_indices_1, train_indices_2])
    test_indices = np.concatenate([test_indices_0, test_indices_1, test_indices_2])
    
    X_train, y_train = X[train_indices], y_one_hot[train_indices]
    X_test, y_test = X[test_indices], y_one_hot[test_indices]
    
    return (X_train, y_train), (X_test, y_test)

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((np.max(y_true) + 1, np.max(y_true) + 1), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    return cm

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100
