# Human-Activity-Recognition
Recognizing human activities using Deep Learning

## Dataset
[Recognition of Human Actions](http://www.nada.kth.se/cvap/actions/)

- [Walking.zip](http://www.nada.kth.se/cvap/actions/walking.zip)
- [Jogging.zip](http://www.nada.kth.se/cvap/actions/jogging.zip)
- [Running.zip](http://www.nada.kth.se/cvap/actions/running.zip)
- [Boxing.zip](http://www.nada.kth.se/cvap/actions/boxing.zip)
- [Handwaving.zip](http://www.nada.kth.se/cvap/actions/handwaving.zip)
- [Handclapping.zip](http://www.nada.kth.se/cvap/actions/handclapping.zip)

```
NOTE:
The following file is corrupted which might give an error when read - 
'person01_boxing_d4_uncomp.avi' (present in 'Boxing.zip')

Either fix the file or delete it before proceeding.
```

## Instructions
1. Clone the repository and navigate to the downloaded folder.

  ```
		git clone https://github.com/MrinalJain17/Human-Activity-Recognition
  ```

2. Unzip the compressed data files and store in the format as mentioned [here](https://github.com/MrinalJain17/Human-Activity-Recognition/blob/master/Directory%20Structure%20for%20Data.txt)
	- Use the helper function `download_files()` present in `data_utils.py` to do this in your current working directory automatically.

3. Run the notebook

  ```
		jupyter notebook human_activity_recognition.ipynb
  ```

## Requirements
`Python 3.x`

#### Libraries:
- [Numpy, Scipy, Jupyter, Ipython, Matplotlib](https://scipy.org/install.html)
- [Scikit-learn](http://scikit-learn.org/stable/install.html)
- [Scikit-video](http://www.scikit-video.org/stable/)
- [Tensorflow](https://www.tensorflow.org/install/)
- [Keras](https://keras.io/#installation)
- [tqdm](https://pypi.python.org/pypi/tqdm#installation)
