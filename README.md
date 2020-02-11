# Chest Xray Classification 

## Data organization	
TODO

## Dataflow implementation 
TODO


## To train the code
```bash
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 
```


## To run the evaluation: turn on the flag eval (evaluation need label to evaluate, model weight, of course)
```bash
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 \
--eval --load=train_log/ResNet101/se/256/5/model-178750.index
```


## To run the prediction: turn on the flag pred (prediction doesnot need label but model weight)
```bash
python run_vinmec.py --gpus='2' --name=ResNet101 --mode=se  --shape=256 --batch=64 \
--pred --load=train_log/ResNet101/se/256/5/model-178750.index
```
