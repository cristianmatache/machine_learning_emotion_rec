# Machine Learning CBC 1 - Emotion Recognition

Implementation of decision tree and decision forest algorithms. Each of these algorithms will be used in order to identify six basic emotions from the human facial expressions (anger, disgust, fear, happiness, sadness and surprise) based on a labelled set of facial Action Units (AUs). AUs correspond to contractions of human facial muscles, which underlie each and every facial expression, including facial expressions of the six basic emotions.

## Getting Started

Install dependency libraries by running:

```
sudo apt-get update
sudo apt-get -y install python3-pip
sudo pip install pipreqs
sudo pip3 install -r requirements.txt
```

It will install pip3 alognside the following libraries: networkx, matplotlib, pandas, scipy, numpy.

## Running the algorithms

We provide a command-line based program. The default command, which will run the decision forest algorithm on the Data/cleandata_students.mat is:

```
python3 main.py
```

The program actually takes between 0 and 2 arguments:

0 -> default version from above

```
python3 main.py
```

1 -> either 'tree' or 'forest', which decides which algorithm you want to apply on the data. The algorithm will run o a single process.
```
python3 main.py tree
python3 main.py forest
```

2 -> the first argument is either tree or forest, as above. The second argument should be either 'single' or 'multi', which will decide whether the chosen algorithm will be run on a single process or on multiple processes for improved computational time.
```
python3 main.py tree single
python3 main.py forest single
python3 main.py tree multi
pythone main.py forest multi
```

## Authors

* **Adrian Catana** 
* **Andrei Isaila** 
* **Cristian Matache** 
* **Oana Ciocioman**