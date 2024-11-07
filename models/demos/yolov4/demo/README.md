To run the demo of yolov4, run the command: `pytest models/demos/yolov4/demo/demo.py`

The current demo uses the coco dataset to test the demo. 500 images of coco-2017 validation data is taken.
To increase the number of samples tested, plese follow the below process:
1. Use the below code to retrieve the data from coco dataset:
```
import fiftyone
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["person", "car"],
    max_samples=500,
)
```
Note:
- Modify the dataset year, split and max_samples based on requirements.
- In Google Colab, the data is saved in the directory "/root/fiftyone/coco-2017/validation"
2. Download the data folder, zip it and upload to google drive. Ensure the access of "Anyone with the link" is given to the uploaded file .
3. Copy the File ID from the drive link of uploaded file and paste it in DATA_PATH_ID (line no. 9)

Note: The .sh file gets exceuted while running the demo file, only when the zip file doesn't exist in the "models/demos/yolov4/demo/" folder. If the demo file is ran already, delete the zip file and data folder to ensure the new data gets downloaded without any issues.
