import requests
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, data_dir: str, api_url: str):
        self.data_dir = Path(data_dir)
        self.rgbd_dir = self.data_dir / 'rgbd'
        self.segmented_dir = self.data_dir / 'segmented'
        self.api_url = api_url

    def clear_directories(self):
        for dir_path in [self.rgbd_dir, self.segmented_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

    def fetch_and_save_data(self, frame_id: str):
        files = {
            'depth_frame': ('depth.raw', b''),  # Empty bytestring as placeholder
            'rgb_frame': ('image.png', b'')     # Empty bytestring as placeholder
        }

        data = {
            'depth_metadata': '{}',  # Empty JSON as placeholder
            'rgb_metadata': '{}'     # Empty JSON as placeholder
        }

        response = requests.post(f"{self.api_url}/getData/{frame_id}", files=files, data=data)
        response.raise_for_status()
        
        response_data = response.json()
        
        depth_bytes = response_data['depth_frame']
        depth_array = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((90, 160))
        depth_path = self.rgbd_dir / f"depth_frame_{frame_id}.raw"
        depth_array.tofile(depth_path)

        with open(self.rgbd_dir / f"depth_frame_{frame_id}.meta", 'w') as f:
            json.dump(response_data['depth_metadata'], f, indent=4)

        rgb_bytes = response_data['rgb_frame']
        rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8)
        rgb_img = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        cv2.imwrite(str(self.segmented_dir / f"rgb_frame_{frame_id}.png"), rgb_img)

        with open(self.segmented_dir / f"rgb_frame_{frame_id}.meta", 'w') as f:
            json.dump(response_data['rgb_metadata'], f, indent=4)
