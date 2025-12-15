import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(json_file):
    """
    Evaluate predictions from a JSON file.
    
    Labels and predictions:
    - 1: supported (positive)
    - 0: not supported (negative)
    - -1: error output (prediction only, always counted as wrong)
    
    When prediction is -1, we treat it as 0 (negative) for metric calculations.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    labels = []
    predictions = []
    
    for item in data:
        label = item['label']
        prediction = item['prediction']
        
        # Treat -1 predictions as 0 (negative/not supported)
        # This ensures they count as wrong when label=1 (False Negative)
        # and wrong when label=0 (since -1 indicates an error)
        if prediction == -1:
            prediction = 0
        
        labels.append(label)
        predictions.append(prediction)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    # Print results
    print(f"Evaluation Results:")
    print(f"=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"=" * 50)
    
    # Additional statistics
    num_errors = sum(1 for item in data if item['prediction'] == -1)
    if num_errors > 0:
        print(f"\nNote: {num_errors} predictions were -1 (errors), treated as negative predictions")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(labels),
        'num_errors': num_errors
    }

if __name__ == "__main__":
    import sys
    
    # Default to baseline_results.json if no argument provided
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'baseline_results.json'
    
    results = evaluate_predictions(json_file)

