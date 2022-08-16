# wf_denoising
Author: Yusuke Uchiyama

This is to denoise waveform data using
a denoising convolutional autoencoder technique.

### Environment

* Python3.6
* Tensorflow2.0 + tf.keras


## Usage

To train and plot  
```
$ ./denoising5_tf2.py 0 1
```
To train without display (job)
```
$ srun --cpus-per-task=10 ./denoising5_tf2.py 0 0
```
or
```
$ sbatch --cpus-per-task=10 ./denoising5_tf2.py 0 0
```


The results will be saved as .h5 files.
The training status and result will be sent to comet.ml server.

#### comet.ml
The api_key and workspace are supposed to be set in a .comet.config file.
Otherwise, set them in the code:
```python
Experiment(api_key="AAAXXX", workspace = "yyy", project_name="wf_denoising")
```

### Input datasets

You need datasets of signal and noise, separately, in pickle format.
The noisy dataset is made inside the program by summing the waveforms in the two files, waveform-by-waveform sequentially.

As the signal (no noise) datasets, the bartender output can be used.
To be insensitive to the position (timing) of signal pulses in the time window,
the data should be made by randomly mixing subevents without putting
any subevents at fixed timings.

As the noise datasets, pedestal run data can be used.


#### To make a pickl file

First, use `read_wf_macro.C` to decode a raw.root file.
```
$ ./meganalyzer -I 'read_wf_macro.C("raw11100.root")'
```
You have to modify some parameters in the macro to select channels to be extracted and to set the size of dataset.

You will get a .csv file.
Since handling csv file is not efficient (e.g. loading a large size csv file takes too long), the script does not accept .csv file but .pkl one. 

Next, convert the csv file to pickle file uisng `csv2pickle.py`
```
$ ./csv2pickle.py wf11100.csv
```
(2022 Aug.) Use pandas version 1.1.4. Pickle file made with 1.4.1 cannot be loaded in Google colab environment. (Both pandas and pickle protocol versions are incompatible.)


#### To convert to ONNX file

```bash
$ ./convert2onnx.py noiseextraction2D8ch_tf2_20220812.h5 CDCHWfDenoising_Method1_402042_20220812_2D8ch.onnx
```
Download the trained Keras model (*.h5) to this directory.
You may have to set python environment up before running the conversion script, like:
```bash
$ conda activate tf2
```
To convert .h5 file, onnxmltools 1.8 is needed; do not update higher version.


## Trial and error

### Architecture


### Filter


### Loss function and activation of output layer
The following were tried and worked

* 'mse' + 'linear'
* 'mse' + 'sigmoid'
* 'binary_crossentropy' + 'sigmoid'

The best seems 'mse' + 'linear'.
'mse' + 'sigmoid' works but the learning speed is slow, as expected.
'binary_crossentropy' + 'sigmoid' also worked. The output is limited to [0,1].
Different loss functions cannot be directly compared and so it is not clear whether there are any differences from 'mse' case.


### Adjustment of scale and offset

The scale and offset of the input waveform seem important.
Regarding to CDCH waveforms, adding offset of 50 mV and scaling up by a factor 5 works well.

Note that if you use the model (.h5 file) in other place, hese parameters have to be applied to the input/output of predict() 
to convert/revert to the original scale.

