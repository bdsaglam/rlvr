#!/usr/bin/env python3
"""
Simple test to verify the MuSiQue environment loads correctly.
"""

import sys
sys.path.insert(0, "environments")
sys.path.insert(0, "src")

def test_environment_loading():
    """Test if the MuSiQue environment can be loaded."""
    try:
        print("🧪 Testing MuSiQue environment loading...")
        
        # Import the load function directly
        from vf_musique import load_environment
        
        print("✅ Successfully imported load_environment")
        
        # Try to load with minimal settings
        print("🌍 Loading environment with minimal settings...")
        env = load_environment(
            num_train_examples=10,  # Small number for testing
            num_eval_examples=5,
            retriever_name="golden",  # Simplest retriever
            judge_model=None,  # Skip judge for now
        )
        
        print(f"✅ Environment loaded successfully!")
        print(f"📊 Train dataset size: {len(env.dataset)}")
        print(f"📊 Eval dataset size: {len(env.eval_dataset) if env.eval_dataset else 'None'}")
        print(f"🔧 Number of tools: {len(env.tools)}")
        print(f"🏷️  Parser type: {type(env.parser).__name__}")
        print(f"📏 Max turns: {env.max_turns}")
        
        # Test tool names
        tool_names = [tool.__name__ for tool in env.tools]
        print(f"🛠️  Available tools: {tool_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_environment_loading()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        sys.exit(1)