### DeepDream-in-PyTorch

!(./example-images/preview3.jpg)
!(./example-images/preview4.jpeg)


Run on Python 3.8 with PyTorch 1.9.0 and CUDA 11.1 (with cuDNN 8)

# Installation via pip
Install requirements inside your virtual environment as such
```bash
pip3 install -r requirements.txt
```
or like this, if the previous line did not work
```bash
cat requirements.txt | cut -f1 -d"#" | sed '/^\s*$/d' | xargs -n 1 pip install
```

## Run

Can be run from command line as such:
```bash
python3 train.py path/to/image.file
```

With arguments `-r N` where N is the number of repeats and `-s` if a single FC-layer neuron should be selected at a time, defaults to a random value, but can be set to an integer value. If forward-passing to the classification layer, a class from labels.txt can be selected.
More can be found in
```bash
python3 train.py --help
```
