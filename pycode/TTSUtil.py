#!pip install playsound
#!pip install gTTS
#!pip install pyobjc

from gtts import gTTS
from playsound import playsound
import os

import IPython
from IPython.core.display import display

def speak(text, ipython=False):
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save("./speak.mp3")
    if ipython:
        display(IPython.display.Audio("./speak.mp3", autoplay=True))
    else:
        playsound("./speak.mp3")
    os.remove("./speak.mp3")
    