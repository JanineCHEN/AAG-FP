# AAG-FP
This project presents a framework for extracting attributed adjacency graphs (AAGs) from floor plan images.

# Dependencies
- Linux or macOS is required
- Python ≥ 3.6
- <a href="https://detectron2.readthedocs.io/en/latest/tutorials/install.html">detectron2<a> # need to be installed separately
- <a href="https://pytorch.org/">pytorch<a> # need to be installed separately
- Other required packages are summarized in `requirements.txt`.

# Quick start
## Download the repository and install the dependencies 
```
git clone https://github.com/JanineCHEN/AAG-FP.git 
cd ~/AAG_FP/
conda create --name AAG_FP python=3.6 # can use either anaconda or virtualenvwrapper to create the virtal environment
conda activate AAG_FP
# detectron2 and pytorch need to be installed separately
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

# Download the data
For downloading the dataset, please refer to <a href="https://github.com/JanineCHEN/AAG-FP/tree/main/dataset">dataset</a>.

# Citation
If you find the code in our research useful, please consider cite:
Coming soon...

# Acknowledgments
Part of the code is inspired by <a href="https://github.com/CubiCasa/CubiCasa5k">CubiCasa/CubiCasa5k</a> and <a href="https://github.com/yu45020/Text_Segmentation_Image_Inpainting">yu45020/Text_Segmentation_Image_Inpainting</a>. The computational work for this project was partially performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg).
