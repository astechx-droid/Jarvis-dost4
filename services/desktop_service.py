import os
import subprocess
import webbrowser
import pyautogui
import psutil
import logging

logger = logging.getLogger(__name__)

def open_application(app_name: str):
    """Opens a common desktop application based on name."""
    app_map = {
        "chrome": "chrome",
        "notepad": "notepad",
        "calculator": "calc",
        "vscode": "code",
        "paint": "mspaint",
        "explorer": "explorer",
    }
    
    name = app_name.lower()
    target = app_map.get(name, name)
    
    print(f"JARVIS: Opening {name}...")
    try:
        subprocess.Popen(target, shell=True)
        return f"Sir, I have opened {app_name} for you."
    except Exception as e:
        logger.error(f"Failed to open {app_name}: {e}")
        return f"Sir, I couldn't open {app_name}. Error details: {str(e)}"

def control_volume(action: str):
    """Controls volume using media keys."""
    if action == "up":
        for _ in range(5): pyautogui.press("volumeup")
        return "Sir, I have increased the volume."
    elif action == "down":
        for _ in range(5): pyautogui.press("volumedown")
        return "Sir, I have decreased the volume."
    elif action == "mute":
        pyautogui.press("volumemute")
        return "Sir, I have muted the system."
    return "Action not recognized, Sir."

def search_local_files(query: str, search_path: str = "C:"):
    """
    Search for files containing the query in their name.
    Limited to a few levels of depth for performance.
    """
    results = []
    print(f"JARVIS: Searching for '{query}' in {search_path}...")
    
    # We restrict depth for safety and speed
    try:
        root_dir = os.path.expanduser(search_path)
        for root, dirs, files in os.walk(root_dir):
            # Limit depth to avoid infinite loops or slow scans
            if root.count(os.sep) - root_dir.count(os.sep) > 3:
                del dirs[:]
                continue
                
            for file in files:
                if query.lower() in file.lower():
                    results.append(os.path.join(root, file))
                    if len(results) > 5: break
            if len(results) > 5: break
            
        if results:
            return "Sir, I found these files:\n" + "\n".join(results)
        return f"Sir, I couldn't find any files matching '{query}' within range."
    except Exception as e:
        return f"Sir, searching failed: {str(e)}"

def system_power(action: str):
    """Shutdown or Restart the system (with 10-second delay for safety)."""
    if action == "shutdown":
        os.system("shutdown /s /t 10")
        return "Sir, the system will shut down in 10 seconds. Save your work."
    elif action == "restart":
        os.system("shutdown /r /t 10")
        return "Sir, the system will restart in 10 seconds."
    return "Action not recognized."

def browser_action(url: str = None, action: str = "open"):
    """Browser automation tasks."""
    if action == "open" and url:
        if not url.startswith("http"): url = "https://" + url
        webbrowser.open(url)
        return f"Sir, the website {url} is now open."
    elif action == "new_tab":
        pyautogui.hotkey("ctrl", "t")
        return "Sir, I've opened a new tab."
    elif action == "close_tab":
        pyautogui.hotkey("ctrl", "w")
        return "Sir, I've closed the current tab."
    return "Sir, I'm not sure what you want to do with the browser."

def get_system_stats():
    """Returns CPU and RAM usage."""
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    return f"Sir, the CPU usage is at {cpu}% and RAM is at {ram}%."
