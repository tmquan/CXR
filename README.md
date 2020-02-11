# Chest Xray Classification 


To train the code
```python
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 
```

```python
To run the evaluation: turn on the flag eval (evaluation need label to evaluate, model weight, of course)
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 --eval --load=train_log/ResNet101/se/256/5/model-178750.index
```

```python
To run the prediction: turn on the flag pred (preduction doesnot need label but model weight)
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 --eval --load=train_log/ResNet101/se/256/5/model-178750.index
```
