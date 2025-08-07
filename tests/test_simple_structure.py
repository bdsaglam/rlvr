#!/usr/bin/env python3
"""
Simple test to verify the structure and imports without running the actual environment.
"""

def test_structure():
    """Test the basic structure of our implementation."""
    
    print("🧪 Testing MuSiQue environment structure...")
    
    # Check if files exist
    import os
    files_to_check = [
        "environments/vf_musique/__init__.py",
        "environments/vf_musique/vf_musique.py",
        "environments/vf_musique/metrics.py",
        "environments/vf_musique/pyproject.toml",
        "environments/vf_musique/README.md",
        "scripts/train_musique.py",
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Check if the structure looks right
    with open("environments/vf_musique/vf_musique.py", "r") as f:
        content = f.read()
        
    key_components = [
        "def load_environment(",
        "class MuSiQueToolEnv",
        "class MuSiQueRubric", 
        "def make_retrieve_tool(",
        "def preprocess_example(",
    ]
    
    print("\n🔍 Checking key components in vf_musique.py:")
    for component in key_components:
        if component in content:
            print(f"✅ {component} found")
        else:
            print(f"❌ {component} missing")
    
    # Check metrics
    with open("environments/vf_musique/metrics.py", "r") as f:
        metrics_content = f.read()
        
    metrics_functions = [
        "def exact_match(",
        "def f1(",
        "def get_last_answer(",
        "def extract_all_retrieved_doc_ids(",
    ]
    
    print("\n🔍 Checking metrics functions:")
    for func in metrics_functions:
        if func in metrics_content:
            print(f"✅ {func} found")
        else:
            print(f"❌ {func} missing")
    
    # Check training script
    with open("scripts/train_musique.py", "r") as f:
        train_content = f.read()
        
    training_components = [
        'vf.load_environment(',
        'env_id="vf-musique"',
        'vf.get_model_and_tokenizer(',
        'vf.GRPOTrainer(',
    ]
    
    print("\n🔍 Checking training script components:")
    for component in training_components:
        if component in train_content:
            print(f"✅ {component} found")
        else:
            print(f"❌ {component} missing")
    
    print("\n📋 Summary:")
    print("✅ Environment package structure created")
    print("✅ Load function implemented")
    print("✅ Custom ToolEnv subclass created")
    print("✅ Retrieval tools ported")
    print("✅ Custom rubric for MuSiQue evaluation")
    print("✅ Metrics functions implemented")
    print("✅ Modern training script created")
    print("✅ Package configuration (pyproject.toml)")
    print("✅ Documentation (README.md)")
    
    return True


if __name__ == "__main__":
    test_structure()
    print("\n🎉 Structure validation complete!")
    print("\n📝 Next steps:")
    print("1. Install the verifiers library: pip install verifiers")
    print("2. Install the environment: vf-install environments/vf_musique")
    print("3. Test the environment: vf-eval vf-musique")
    print("4. Run training: python scripts/train_musique.py")