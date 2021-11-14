#Install stuff
#conda create -n img_class_ml_env python=3.7
#conda activate img_class_ml_env
pip install --upgrade pip

pip install opencv-python==4.5.3.*
pip install tensorflow==2.7.*
pip install Pillow
pip install playsound
pip install gTTS

pip install matplotlib
pip install pandas
pip install seaborn

pip install PyYAML

pip list --format=freeze > requirements.txt

#pip install ipykernel
#python -m ipykernel install --user --name img_class_ml_env --display-name "img_class_ml_env"