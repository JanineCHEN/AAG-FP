# AAG-FP
This project presents a framework for extracting attributed adjacency graphs (AAGs) from floor plan images.

# Dependencies
- Linux environment is recommended
- Python 3.6 or 3.7
- <a href="https://detectron2.readthedocs.io/en/latest/tutorials/install.html">detectron2<a>
- opencv-python
- scikit-image
- pytorch
- torchvision
- networkx
- Other required packages are summarized in `requirements.txt`.

# Quick start
## Download the repository and install the dependencies 
```
git clone https://github.com/JanineCHEN/AAG-FP.git 
cd ~/AAG_FP/
conda create --name AAG_FP python=3.7
conda activate AAG_FP
pip install -r requirements.txt
```

## Download the checkpoint
For downloading the checkpoints, please refer to <a href="https://github.com/JanineCHEN/AAG-FP/tree/main/ckpt">ckpt</a>.

## Run the framework
This demo example uses the sample floor plan images in `FP_sample_images`.

you can use your own floor plan images by putting them inside the `FP_sample_images` folder, images with extension ".jpeg",".jpg" or ".png" are all accepted.

For executing the AAG extractor, please run:
```
python main.py
```

## Download the data
For downloading the dataset, please refer to <a href="https://github.com/JanineCHEN/AAG-FP/tree/main/dataset">dataset</a>.

# Citation
If you find the code in our research useful, please consider cite:
Coming soon...
