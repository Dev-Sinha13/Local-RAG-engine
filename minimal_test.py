try:
    import rusty_rag
    print(f"Successfully imported rusty_rag from {rusty_rag.__file__}")
    
    tokens = rusty_rag.tokenize("Hello World!")
    print(f"Tokenize test: {tokens}")
    if tokens == ["hello", "world"]:
        print("PASS")
    else:
        print("FAIL (tokens mismatch)")
except ImportError as e:
    print(f"FAIL: ImportError - {e}")
except Exception as e:
    print(f"FAIL: {e}")
