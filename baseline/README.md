## Reproducibility

The comparison result can be done by the following commands, taking the STRING network as an example:

- EMOGI
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_pc50.pkl --model EMOGI --epochs 200 --lr 0.01 --device 0
```

- MTGCN
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_pc50.pkl --model MTGCN --epochs 200 --lr 0.01 --device 0
```

- GCN
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_pc50.pkl --model GCN --epochs 200 --lr 0.01 --device 0
```

- GAT
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_pc50.pkl --model GAT --epochs 200 --lr 0.01 --device 0
```

- SVM
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_pc50.pkl --model SVM --epochs 200 --lr 0.01 --device 0
```

- deepwalk
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_deepwalk_feature.pkl --model SVM --epochs 200 --lr 0.01 --device 0
```

- XGEP
```
python run_baselines.py --is_5_CV_test True --dataset_file ./data/STRING_xgep_feature.pkl --model SVM --epochs 200 --lr 0.01 --device 0
```

- DeepHE

See [DeepHE](https://github.com/xzhang2016/DeepHE/tree/master) for detail explanation.
```
python main.py --embedF 3 --result_dir ./string_pc50 --numHiddenLayer 3
```

### With your own data
We also provide a script `build_baseline_dataset_container.py` to generate a specific dataset
