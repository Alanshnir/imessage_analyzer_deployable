import subprocess
import webbrowser
import time
import sys
import os
import socket

def is_port_open(host='localhost', port=8501, timeout=1):
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    print("Starting iMessage Analyzer...")
    
    # Path to app.py inside the same folder
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Launch Streamlit
    print("Launching Streamlit server...")
    
    # Use streamlit command directly (works regardless of which Python)
    proc = subprocess.Popen(
        ["streamlit", "run", app_path],
        # stdout and stderr will show in terminal
    )

    # Wait for Streamlit to be ready (check if port is open)
    print("Waiting for Streamlit to start...")
    max_wait = 30  # Maximum 30 seconds
    waited = 0
    
    while waited < max_wait:
        if is_port_open('localhost', 8501):
            print("✓ Streamlit is ready!")
            break
        time.sleep(1)
        waited += 1
        if waited % 5 == 0:
            print(f"  Still waiting... ({waited}s)")
    
    if waited >= max_wait:
        print("⚠ Streamlit didn't start in time, but trying to open browser anyway...")
    
    # Open browser to Streamlit UI
    print("Opening browser to http://localhost:8501")
    webbrowser.open("http://localhost:8501")
    
    print("\n✓ iMessage Analyzer is running!")
    print("  Browser should open automatically.")
    print("  If not, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the analyzer.\n")

    # Wait until the Streamlit server stops
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping iMessage Analyzer...")
        proc.terminate()
        proc.wait()
        print("✓ Stopped.")

if __name__ == "__main__":
    main()

