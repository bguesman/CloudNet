# CloudNet
Generative model for clouds based on satellite data.

Download the dataset here: [https://drive.google.com/drive/folders/1mYtuikwmKA5-__ZVxPN3stTrd3sgKlk1?usp=sharing](https://drive.google.com/drive/folders/1mYtuikwmKA5-__ZVxPN3stTrd3sgKlk1?usp=sharing)

You'll first want to set up and activate the virtual environment by running:
```
./setup-env.sh
source cloud-net-env/bin/activate
```

Then run `preprocess.py`, which accepts an input and output path, and processes/writes out the data.
```
python preprocess.py <train data in path> <processed train data out path>
```

You should now be able to run `model_runner.py`, which accepts a training data path and an output path.
```
python model_runner.py <processed train data in path> <results out path>
```