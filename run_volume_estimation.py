from flask import Flask, jsonify
import json
import requests
from pathlib import Path
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.reconstruction.volume_calculator import VolumeCalculator
from src.utils.visualization_3d import Visualizer3D
from src.utils.logging_utils import setup_logging

app = Flask(__name__)
logger = setup_logging(log_dir="logs", log_prefix="volume_estimation")

# Load configuration from file
CONFIG_PATH = Path("test_config.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found.")

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# External Nutrition API URL
NUTRITION_API_URL = "https://starfish-app-fycwd.ondigitalocean.app/api/nutrition"

def process_frames(config):
    results = []
    pipeline = PreprocessingPipeline(config)
    calc = VolumeCalculator(
        camera_height=config['camera_height'],
        plate_diameter=config['plate_diameter']
    )
    
    for frame_id in config['frame_ids']:
        result = pipeline.process_single_image(frame_id)
        plate_data = next(
            (data for name, data in result['processed_objects'].items() if name == 'plate'),
            None
        )
        if plate_data is None:
            raise ValueError("No plate found in processed objects")
        
        calibration = calc.calculate_plate_reference(
            depth_map=result['depth'],
            plate_mask=plate_data['mask'],
            intrinsic_params=result['intrinsic_params']
        )
        plate_height = calibration['plate_height']
        
        volume_results = {}
        for obj_name, obj_data in result['processed_objects'].items():
            if obj_name == 'plate':
                continue
            volume_data = calc.calculate_volume(
                depth_map=result['depth'],
                mask=obj_data['mask'],
                plate_height=plate_height,
                intrinsic_params=result['intrinsic_params'],
                calibration=calibration
            )
            volume_results[obj_name] = volume_data

        results.append({
            "frame_id": frame_id,
            "volumes": [
                {
                    "object_name": obj_name,
                    "volume_cups": volume_data['volume_cups'],
                    "uncertainty_cups": volume_data['uncertainty_cups']
                }
                for obj_name, volume_data in volume_results.items()
            ]
        })
    return results

def fetch_nutrition_data(volumes):
    # Prepare request data for nutrition API
    nutrition_request = {
        "data": [
            {
                "food_name": obj['object_name'],
                "volume": obj['volume_cups']
            }
            for obj in volumes
        ]
    }
    
    try:
        # Make a POST request to the nutrition API
        response = requests.post(NUTRITION_API_URL, json=nutrition_request)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch nutrition data: {e}", exc_info=True)
        return {"error": "Unable to fetch nutrition data"}

@app.route('/get_volumes', methods=['GET'])
def get_volumes():
    try:
        # Process frames and calculate volumes
        frame_results = process_frames(CONFIG)
        
        for frame in frame_results:
            # Fetch nutrition data for each frame
            nutrition_data = fetch_nutrition_data(frame['volumes'])
            # Add nutrition data to the frame
            frame['nutrition'] = nutrition_data
        
        return jsonify({"results": frame_results}), 200
    except Exception as e:
        logger.error(f"Failed to process: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

