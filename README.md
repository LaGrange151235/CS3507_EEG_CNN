This repository is for the course project of CS3507 Engineering Practice and Technological Innovation IV-J. I have implemented 2 CNN models (a naive CNN and ResNet18) for a emotion recognition task over [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) dataset.

---
## Install
```
pip install -r requirements.txt
```

## Run
```
python train.py --lr [learning rate] --batch_size [batch size] --num_epochs [number of epochs] --mode [dependent/independent] --model [CNN/ResNet18]
```
- Dependent train: depend on the subjects, which means we will train $15\times 3=45$ models for each subject each session.
- Independent train: independ from the subjects, which means we will train with data from $14$ subjects and test with another $1$ subject, and get $15$ models in total.