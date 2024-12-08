# M-PCAM Model

## Getting started
  - start a virtual environment
  
  to run the system ``` python run_volume_estimation.py``` 
  


## shit to improve on
- initialize the flask app
- add function to call api for the macronutrient database.




# TO RUN 

python run_volume_estimation --config test_config.json

endpoint is http://127.0.0.1:5000/get_volumes


{
  "results": [
    {
      "frame_id": "frame_1",
      "volumes": [
        {
          "object_name": "egg scrambled",
          "volume_cups": 0.5,
          "uncertainty_cups": 0.05
        },
        {
          "object_name": "oatmeal",
          "volume_cups": 1.0,
          "uncertainty_cups": 0.1
        }
      ],
      "nutrition": {
        "data": [
          {
            "food_name": "egg scrambled",
            "calories": 70,
            "protein": 6,
            "carbs": 1,
            "fat": 5
          },
          {
            "food_name": "oatmeal",
            "calories": 150,
            "protein": 5,
            "carbs": 27,
            "fat": 3
          }
        ]
      }
    }
  ]
}

