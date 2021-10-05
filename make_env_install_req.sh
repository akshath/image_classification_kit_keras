#create env
conda create -n img_class_ml_env python=3.8

#activate
conda activate img_class_ml_env

pip install --upgrade pip

pip install opencv-python
pip install tensorflow
#tensorflow 2.6.0 needs numpy 1.19.5
#pip install numpy==1.19.5

pip install Pillow
pip install playsound
pip install gTTS
#for mac
pip install pyobjc

pip install matplotlib
pip install pandas
pip install seaborn

pip install PyYAML

#so we can access this env in Jupiter notebook
pip install ipykernel
python -m ipykernel install --user --name img_class_ml_env --display-name img_class_ml_env
