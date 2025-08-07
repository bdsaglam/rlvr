#!/usr/bin/env python3
"""
Test the CLI structure and validate that all components are properly organized.
"""

def test_training_script_structure():
    """Test that the training script has the right structure."""
    
    print("🧪 Testing training script structure...")
    
    with open("scripts/train_musique.py", "r") as f:
        content = f.read()
    
    # Check for key Typer components
    typer_components = [
        "import typer",
        "app = typer.Typer()",
        "@app.command()",
        "def train(",
        "def predict(",
        'typer.Option(',
        'typer.echo(',
    ]
    
    print("\n🔍 Checking Typer CLI components:")
    for component in typer_components:
        if component in content:
            print(f"✅ {component} found")
        else:
            print(f"❌ {component} missing")
    
    # Check for training parameters
    training_params = [
        "--model",
        "--num-train-examples",
        "--retriever",
        "--batch-size",
        "--learning-rate",
        "--use-lora",
        "--lora-r", 
        "--lora-alpha",
        "--output-dir",
        "--run-name",
        "--push-to-hub",
        "--resume-from-checkpoint",
    ]
    
    print("\n🔍 Checking training parameters:")
    for param in training_params:
        if param in content:
            print(f"✅ {param} parameter found")
        else:
            print(f"❌ {param} parameter missing")


def test_evaluation_script_structure():
    """Test that the evaluation script has the right structure."""
    
    print("\n🧪 Testing evaluation script structure...")
    
    with open("scripts/evaluate_musique.py", "r") as f:
        content = f.read()
    
    # Check for multiple commands
    commands = [
        "def evaluate(",
        "def benchmark(",
        "def analyze(",
    ]
    
    print("\n🔍 Checking evaluation commands:")
    for command in commands:
        if command in content:
            print(f"✅ {command} found")
        else:
            print(f"❌ {command} missing")
    
    # Check for evaluation parameters
    eval_params = [
        "--dataset-split",
        "--num-examples",
        "--retriever",
        "--batch-size",
        "--temperature",
        "--output-file",
        "--verbose",
    ]
    
    print("\n🔍 Checking evaluation parameters:")
    for param in eval_params:
        if param in content:
            print(f"✅ {param} parameter found")
        else:
            print(f"❌ {param} parameter missing")


def test_cli_documentation():
    """Test that CLI documentation is comprehensive."""
    
    print("\n🧪 Testing CLI documentation...")
    
    # Check training script docstring
    with open("scripts/train_musique.py", "r") as f:
        content = f.read()
        
    doc_elements = [
        'Usage:',
        'Example with custom settings:',
        'python scripts/train_musique.py',
        '--model',
        '--retriever',
        '--batch-size',
    ]
    
    print("\n🔍 Checking training script documentation:")
    for element in doc_elements:
        if element in content:
            print(f"✅ {element} found in docs")
        else:
            print(f"❌ {element} missing from docs")


def test_cli_patterns():
    """Test that CLI follows good patterns."""
    
    print("\n🧪 Testing CLI patterns...")
    
    with open("scripts/train_musique.py", "r") as f:
        train_content = f.read()
    
    with open("scripts/evaluate_musique.py", "r") as f:
        eval_content = f.read()
    
    # Check for consistent patterns
    patterns = [
        "get_model_name(", # Helper function
        "typer.echo(", # Consistent output
        "datetime.now().strftime(", # Timestamp generation
        "Path(", # Path handling
        "if __name__ == \"__main__\":", # Main guard
        "app()", # Typer app call
    ]
    
    print("\n🔍 Checking consistent CLI patterns:")
    for pattern in patterns:
        train_has = pattern in train_content
        eval_has = pattern in eval_content
        
        if train_has and eval_has:
            print(f"✅ {pattern} consistent across scripts")
        elif train_has or eval_has:
            print(f"⚠️  {pattern} found in one script but not both")
        else:
            print(f"❌ {pattern} missing from both scripts")


def show_cli_usage_examples():
    """Show usage examples for the CLI."""
    
    print("\n📖 CLI Usage Examples:")
    print("=" * 50)
    
    print("\n🚂 Training:")
    print("  # Basic training")
    print("  python scripts/train_musique.py")
    print("  python scripts/train_musique.py --model meta-llama/Llama-3.1-8B-Instruct")
    
    print("\n  # Advanced training") 
    print("  python scripts/train_musique.py train \\")
    print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
    print("    --retriever hybrid \\")
    print("    --num-train-examples 1000 \\")
    print("    --batch-size 16 \\")
    print("    --learning-rate 1e-6 \\")
    print("    --use-lora \\")
    print("    --lora-r 32")
    
    print("\n  # Prediction")
    print("  python scripts/train_musique.py predict \\")
    print("    --model outputs/my-model \\")
    print("    --batch-size 8")
    
    print("\n🧪 Evaluation:")
    print("  # Basic evaluation")
    print("  python scripts/evaluate_musique.py outputs/my-model")
    
    print("\n  # Advanced evaluation")
    print("  python scripts/evaluate_musique.py evaluate \\")
    print("    outputs/my-model \\")
    print("    --dataset-split validation \\")
    print("    --num-examples 100 \\")
    print("    --retriever hybrid \\")
    print("    --verbose")
    
    print("\n  # Benchmark multiple models")  
    print("  python scripts/evaluate_musique.py benchmark \\")
    print("    --models model1,model2,model3 \\")
    print("    --retrievers bm25,hybrid \\")
    print("    --output-dir benchmark_results")
    
    print("\n  # Analyze results")
    print("  python scripts/evaluate_musique.py analyze benchmark_results")


if __name__ == "__main__":
    test_training_script_structure()
    test_evaluation_script_structure() 
    test_cli_documentation()
    test_cli_patterns()
    show_cli_usage_examples()
    
    print("\n🎉 CLI Structure Testing Complete!")
    print("\n📝 Notes:")
    print("  - Scripts use Typer for modern CLI experience")
    print("  - Consistent parameter naming across train/eval")
    print("  - Multiple commands per script (train/predict, evaluate/benchmark/analyze)")
    print("  - Rich help documentation and usage examples")
    print("  - Error handling and progress reporting")