import openwakeword
import os

print(f"Package location: {os.path.dirname(openwakeword.__file__)}")

try:
    from openwakeword.model import Model
    print("Model class imported successfully")
except Exception as e:
    print(f"Error importing Model: {e}")

# Check resources
resources_path = os.path.join(os.path.dirname(openwakeword.__file__), "resources")
print(f"Resources path exists: {os.path.exists(resources_path)}")

if os.path.exists(resources_path):
    print(f"Contents of resources: {os.listdir(resources_path)}")
    models_path = os.path.join(resources_path, "models")
    print(f"Models path exists: {os.path.exists(models_path)}")
    if os.path.exists(models_path):
        print(f"Available models: {os.listdir(models_path)}")
