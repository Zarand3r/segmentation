import json

data = {}  
data["inception_v3_2016_08_28"] = {
    "input_height": 299,
    "input_width": 299,
    "model_file": "inception_v3_2016_08_28/inception_v3_2016_08_28_frozen.pb",
    "script": "",
    "checkpoint_file": "",
    "label_file": "inception_v3_2016_08_28/labels.txt",
    "input_node": "input",
    "output_node": "InceptionV3/Predictions/Reshape_1"
}  


# mobilenet_v1_1.0_224

with open("models.json", "w") as outfile:  
    json.dump(data, outfile)