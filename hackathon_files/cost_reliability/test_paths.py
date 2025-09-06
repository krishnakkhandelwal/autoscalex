# cost_reliability/test_paths.py
import os

def test_paths():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(base_dir)
    model_dir = os.path.join(parent_dir, 'models')

    print(f"Base directory: {base_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Model directory: {model_dir}")
    print("Exists:", os.path.exists(model_dir))
    print("Files in model_dir:")
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            print(" -", f)

    expected_model = os.path.join(model_dir, 'xgb_scaling_nexth.pkl')
    print(f"Expected model path: {expected_model}")
    print("Model exists:", os.path.exists(expected_model))

if __name__ == "__main__":
    test_paths()
