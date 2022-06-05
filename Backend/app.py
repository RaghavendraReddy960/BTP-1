from crypt import methods
from dataclasses import dataclass
from email.errors import FirstHeaderLineIsContinuationDefect
from fileinput import filename
from pickle import TRUE
from flask import Flask,render_template,request,flash,redirect,jsonify
import numpy as np
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from PIL import Image
import time
import os



def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

#pulse main function
def calc_pauses(audio_path):
  values = []
  x_range = []
  snd = parselmouth.Sound(audio_path)
  data = snd.values.T
  cnt = 0
  for i in range(0 , len(data) , 2205):
    stf = data[i][0]**2
    zcr = 0
    cnt+=1
    for j in range(i+1 , i+2205):
      if j>=len(data):
        break
      stf += data[j][0]**2
      if data[j][0]*data[j-1][0] < 0:
        zcr+=1
    
    
    stf = stf
    
    final_val = stf*100/zcr
    x_range.append(cnt)
    values.append(final_val)
  return values , x_range

#pause Healthy
def pulse_H(f):
  filename = f
  y_values , x_values = calc_pauses(filename)
  plt.figure()
  plt.plot(x_values,y_values)
  plt.xlabel("frameNumber")
  plt.ylabel("ste/zcr")
  plt.savefig("pau_h1")

#pause parkinson
def pulse_p(f):
  filename = f
  y_values , x_values = calc_pauses(filename)
  plt.figure()
  plt.plot(x_values,y_values)
  plt.xlabel("frameNumber")
  plt.ylabel("ste/zcr")
  plt.savefig("pau_p1")

#pause test
def pulse_t(f):
  filename = f
  y_values , x_values = calc_pauses(filename)
  plt.figure()
  plt.plot(x_values,y_values)
  plt.xlabel("frameNumber")
  plt.ylabel("ste/zcr")
  plt.savefig("pau_t")


#amplitude Healthy
def First_function_1(f):
    filename = f 
    snd = parselmouth.Sound(filename)
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.title("amplitude")
    plt.savefig("amp_h1")
    return("execution is done")

#amplitude parkinson
def First_function_2(f):
    snd = parselmouth.Sound(f)
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.title("amplitude")
    plt.savefig("amp_p1")
    return("execution is done")

#amplitude test
def First_function_3(f):
    snd = parselmouth.Sound(f)
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.title("amplitude")
    plt.savefig("amp_t")
    return("execution is done")

#amplitude Healthy
def intensity_1(f):
  snd = parselmouth.Sound(f)
  intensity = snd.to_intensity()
  spectrogram = snd.to_spectrogram()
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_intensity(intensity)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title("intensity")
  plt.savefig("int_h1")

#amplitude parkinson
def intensity_2(f):
  snd = parselmouth.Sound(f)
  intensity = snd.to_intensity()
  spectrogram = snd.to_spectrogram()
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_intensity(intensity)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title("intensity")
  plt.savefig("int_p1")

#amplitude test
def intensity_3(f):
  snd = parselmouth.Sound(f)
  intensity = snd.to_intensity()
  spectrogram = snd.to_spectrogram()
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_intensity(intensity)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title("intensity")
  plt.savefig("int_t")


#pitch Heatlthy
def pitch_1(f):
  snd = parselmouth.Sound(f)
  pitch = snd.to_pitch()
  pre_emphasized_snd = snd.copy()
  pre_emphasized_snd.pre_emphasize()
  spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_pitch(pitch)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title('Pitch')
  plt.savefig("pch_h1")


#pitch parkinson
def pitch_2(f):
  snd = parselmouth.Sound(f)
  pitch = snd.to_pitch()
  pre_emphasized_snd = snd.copy()
  pre_emphasized_snd.pre_emphasize()
  spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_pitch(pitch)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title("pitch")
  plt.savefig("pch_p1")

#pitch test
def pitch_3(f):
  snd = parselmouth.Sound(f)
  pitch = snd.to_pitch()
  pre_emphasized_snd = snd.copy()
  pre_emphasized_snd.pre_emphasize()
  spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
  plt.figure()
  draw_spectrogram(spectrogram)
  plt.twinx()
  draw_pitch(pitch)
  plt.xlim([snd.xmin, snd.xmax])
  plt.title("pitch")
  plt.savefig("pch_t")
#image resize
def resize_c():
  #amp
  img_a_h = Image.open('./amp_h1.png')
  img_a_h = img_a_h.resize((500, 290), Image.ANTIALIAS)
  img_a_h.save('./amp_h.png')
  img_a_p = Image.open('./amp_p1.png')
  img_a_p = img_a_p.resize((500, 290), Image.ANTIALIAS)
  img_a_p.save('./amp_p.png')
  #pitch
  img_p_h = Image.open('./pch_h1.png')
  img_p_h = img_p_h.resize((500, 290), Image.ANTIALIAS)
  img_p_h.save('./pch_h.png')
  img_p_p = Image.open('./pch_p1.png')
  img_p_p = img_p_p.resize((500, 290), Image.ANTIALIAS)
  img_p_p.save('./pch_p.png')
  #intensity
  img_i_h = Image.open('./int_h1.png')
  img_i_h = img_i_h.resize((500, 290), Image.ANTIALIAS)
  img_i_h.save('./int_h.png')
  img_i_p = Image.open('./int_p1.png')
  img_i_p = img_i_p.resize((500, 290), Image.ANTIALIAS)
  img_i_p.save('./int_p.png')
  #pause
  img_pu_h = Image.open('./pau_h1.png')
  img_pu_h = img_pu_h.resize((500, 290), Image.ANTIALIAS)
  img_pu_h.save('./pau_h.png')
  img_pu_p = Image.open('./pau_p1.png')
  img_pu_p = img_pu_p.resize((500, 290), Image.ANTIALIAS)
  img_pu_p.save('./pau_p.png')

#function to remove the images for comparision
def remove():
  os.remove('./amp_h.png')
  os.remove('./amp_h1.png')
  os.remove('./amp_p.png')
  os.remove('./amp_p1.png')
  os.remove('./int_h.png')
  os.remove('./int_h1.png')
  os.remove('./int_p.png')
  os.remove('./int_p1.png')
  os.remove('./pau_h.png')
  os.remove('./pau_h1.png')
  os.remove('./pau_p.png')
  os.remove('./pau_p1.png')
  os.remove('./pch_h.png')
  os.remove('./pch_h1.png')
  os.remove('./pch_p.png')
  os.remove('./pch_p1.png')

#function to remove the images for test single file upload
def remove_1():
  os.remove('./amp_t.png')
  os.remove('./int_t.png')
  os.remove('./pau_t.png')
  os.remove('./pch_t.png')

#flask server  
app = Flask(__name__)

@app.route('/abc', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['hc1']
        file_1 = request.files['pd1']
        filename = secure_filename(file.filename)
        filename_1 = secure_filename(file_1.filename)
        First_function_1(filename)
        First_function_2(filename_1)
        pitch_1(filename)
        pitch_2(filename_1)
        intensity_1(filename)
        intensity_2(filename_1)
        pulse_H(filename)
        pulse_p(filename_1)
        time.sleep(0.5)
        resize_c()
        time.sleep(1)
        remove()
    return ("execution is done")

@app.route('/abc1', methods=['GET', 'POST'])
def upload_file_1():
    if request.method == 'POST':
        file = request.files['htest']
        filename = secure_filename(file.filename)
        First_function_3(filename)
        pitch_3(filename)
        intensity_3(filename)
        pulse_t(filename)
        time.sleep(0.5)
        remove_1()
    return ("execution is done")





if __name__ == "__main__":
    app.run(debug=TRUE)

