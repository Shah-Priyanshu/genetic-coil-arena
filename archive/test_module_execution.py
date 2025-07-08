#!/usr/bin/env python3
"""Test script to debug where API module execution fails."""

import sys

def test_module_execution():
    """Test importing the API module and see where it fails."""
    # Redirect import errors to see the actual issue
    import importlib
    import traceback
    
    try:
        print("Attempting to import API module...")
        
        # Import with full error visibility
        import src.genetic_coil_arena.web.api
        print("Module import completed")
        
        # Check what was actually loaded
        module = sys.modules['src.genetic_coil_arena.web.api']
        print(f"Module file: {module.__file__}")
        print(f"Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
        
        # Try to access the router specifically
        try:
            router = getattr(module, 'router', None)
            if router:
                print(f"Router found: {type(router)}")
            else:
                print("Router not found")
        except Exception as e:
            print(f"Error accessing router: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Module import failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_module_execution()
