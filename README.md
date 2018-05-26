# whoami
Funny face detection project, a good beginning to learn AI. I implement three detection cases: image face detection, live video face detection, and video files face detection. Please enjoy it!

## Requirements
 - tensorflow-gpu
 - python3
 - numpy
 - dlib
 - opencv

## Installation
 `pip install -r requirements.txt`

## Collection your face dataset
In order to detect 'who you are', run `python3 get_my_faces.py` firstly, this code will generate 10000 figures with the form of "label_number.jpg".
Two files `label.txt` and `face_data.txt` will be saved in local folder.

## Image Face Detection
run `python3 image_detection.py`. Note that this step may run very quickly, you can add the delay in the cv2.imshow loop.

## Live Video Face Detection
run `python3 live_video_detection.py`. Note that you should run the image face detection firstly, because this part depends on the `label.txt` and `face_data.txt`. Otherwise, you can creat two fake files to make video detection work well, in this case you will always detect `other` label.

## Video File Face Detection
run `video_file_detection.py`. As above, you should also run the image face detection firstly.

## Results
Because I only labeled `Chandler`, others will be detected as `other`.
![Chandler](https://github.com/mashoujiang/whoami/blob/master/results/Chandler.png)
![Chandler_other](https://github.com/mashoujiang/whoami/blob/master/results/Chandler_other.png)
## Future Work
- Now that we detect faces by calculating Euclidean distance, this is not a good idea due to depending `label.txt` and `face_data.txt`. I will implement this function with neural network later.
