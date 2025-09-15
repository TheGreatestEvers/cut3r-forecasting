# Compile CUDA kernels
cd cut3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../

# Get model weights
pip install gdown
cd cut3r/src
# for 512 dpt ckpt
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../..