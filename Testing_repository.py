#Testing
#!/usr/bin/env python3
"""
Simple Python test file for GitHub repository setup
"""

def greet(name="World"):
    """Simple greeting function"""
    return f"Hello, {name}!"

def add_numbers(a, b):
    """Add two numbers together"""
    return a + b

def main():
    """Main function to test basic functionality"""
    print("🐍 Python Repository Test")
    print("=" * 30)
    
    # Test greeting function
    print(greet())
    print(greet("GitHub"))
    
    # Test math function
    result = add_numbers(5, 3)
    print(f"5 + 3 = {result}")
    
    # Show Python version
    import sys
    print(f"Python version: {sys.version}")
    
    print("\n✅ Repository test completed successfully!")

if __name__ == "__main__":
    main()


