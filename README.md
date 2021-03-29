# face-mask-detection

<a href="https://youtu.be/EPBnpKRRuG0"><img src="markdown/demo-video.png"></a>

## TODO 

- [ ] Build model with diffirent frame-work.

    - [x] Tensorflow.

    - [ ] Pytorch.

    - [ ] MXNet.

- [x] Code preprocessing data.

- [x] Providing preprocessed data.

- [x] Demo .

    - [x] Demo with image.

    - [x] Demo with video.

- [ ] Providing pre-train model.

    - [x] pre-train tensorflow model

    - [ ] pre-train pytorch model

    - [ ] pre-train mxnet model

## Setup environment

```
    pip install -r reqiurements.txt
```

Set variable *USE_GPU* in configs/config.py True if you want to use gpu

## Folder Structure

```
FACE-MASK-DETECION
├──data
|    ├── RMFD
|    |     ├── *jpg
|    ├── RMFD_resize
|    |     ├── *jpg   
|    |
├── images                          # put images you want to test here
│    ├── demo_image.jpg   
|    
├── src
|    ├── *.py
|
├── models                           # put pre-train models here
|    ├── *.h5
|    ├── *.pth
|
├── video                           # put videos your want to test here
|    ├── *.mp4
|
├── .gitignore
├── README.md
├── LICENSE
├── preprocess.py               # run this file first
├── demo_image.py
├── demo_video.py

```




