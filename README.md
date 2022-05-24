# UGaitNet: Multimodal gait recognition with missing input modalities

Support code for paper accepted for publication at IEEE Transactions on Information Forensics and Security.

This work extends [https://github.com/avagait/gaitmiss](https://github.com/avagait/gaitmiss)

## Abstract
Gait recognition systems typically rely solely on silhouettes for extracting gait signatures. Nevertheless, these approaches struggle with changes in body shape and dynamic backgrounds; a problem that can be alleviated by learning from multiple modalities. However, in many real-life systems some modalities can be missing, and therefore most existing multimodal frameworks fail to cope with missing modalities. To tackle this problem, in this work, we propose UGaitNet, a unifying framework for gait recognition, robust to missing modalities. UGaitNet handles and mingles various types and combinations of input modalities, i.e. pixel gray value, optical flow, depth maps, and silhouettes, while being camera agnostic. We evaluate UGaitNet on two public datasets for gait recognition: CASIA-B and TUM-GAID, and show that it obtains compact and state-of-the-art gait descriptors when leveraging multiple or missing modalities. Finally, we show that UGaitNet with optical flow and grayscale inputs achieves almost perfect (98.9%) recognition accuracy on CASIA-B (same-view “normal”) and 100% on TUM-GAID (“ellapsed time”). 

## Proposed architecture

Input: (OF?, Silhouette?, Depth? ellipses) binary input units indicating whether the modality is available – here, depth is not available (dashed red cross); (volumes) sequences of _L_ frames for the different modalities. After fusing the single-modality signatures, a multimodal gait signature of _d_ dimensions is further compressed by FC1. The final FC2 contains _C_ classes (used just for training).  
The proposed model is depicted in the following figure:  
<p  align="center"><img src="images/ugaitnet_tifs.png"></p>

At training, the network learns multimodal signatures so that the distance _D_ between a pair of signatures of the same subject is lower than the distance between signatures of different subjects, independently of the modalities used to generate the signatures. To imitate test situations, some modalities are disabled (i.e. missing) at training (empty shapes).

<p align='center'><img src="images/gaitmiss_loss.png"></p>

## Code

### Requires

* TensorFlow 2.3
* Data files obtained with [gaitutils](https://github.com/avagait/gaitutils) library

### TUM-GAID
#### 1. Training
- Missing modalities:
```python
python mains/mj_trainUWYHGaitNet_DataGen_3mods.py --dbbasedir=DATAPATH/TUM_GAID_tf/ --experdir=OUTPATH --infodir DATAPATH/tumgaid_info/  --datadir DATAPATH/TUM_GAID_tf/ --prefix=ugaitnet_tum --nclasses=150 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=24 --extraepochs=25 --datatype=2 --gaitset --repetitions=5 --mergefun=sign_max
```

- One modality (BL-single experiments in the paper):
```python
python mains/mj_trainUWYHGaitNet_DataGen_1mod.py --dbbasedir=DATAPATH/TUM_GAID_tf/ --experdir=OUTPATH --infodir DATAPATH/tumgaid_info/  --datadir DATAPATH/TUM_GAID_tf/ --prefix=ugaitnet_blsingle_tum --nclasses=150 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=24 --extraepochs=25 --datatype=2 --gaitset --repetitions=5 --mergefun=sign_max
```

- Early fusion (BL-all experiments in the paper):
```python
python mains/mj_trainUWYHGaitNet_DataGen_3mods.py --dbbasedir=DATAPATH/TUM_GAID_tf/ --experdir=OUTPATH --infodir DATAPATH/tumgaid_info/  --datadir DATAPATH/TUM_GAID_tf/ --prefix=ugaitnet_blall_tum --nclasses=150 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=24 --extraepochs=25 --datatype=2 --gaitset --repetitions=5 --mergefun=sign_max --nomissing
```

#### 2. Testing in an open world scenario
```python
python mains/mj_testUWYHGaitNet_open_tum_3mods.py --datadir DATAPATH/tfimdb_tum_gaid_N155_test_s01-02_of25_60x60 --datadirtrain DATAPATH/TUM_GAID_tf/tfimdb_tum_gaid_N155_ft_of25_60x60 --usemod1=1 --usemod2=1 --usemod3=1 --modality gray --modality0 of --verbose 1 --useavg=1 --bs=64 --knn=3 --usemirror 0 --typecode=3 --gaitset --nclasses 155 --model MODELPATH/model-state-0074.hdf5
```

### CASIA-B
#### 1. Training
- Missing modalities:
```python
python mains/mj_trainUWYHGaitNet_DataGen_CasiaB.py --dbbasedir=DATAPATH/CASIAB_tf/ --experdir=OUTPATH --infodir DATAPATH/casiab_info/  --datadir DATAPATH/CASIAB_tf/ --prefix=ugaitnet_casia --nclasses=74 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=40 --extraepochs=25 --datatype=2 --gaitset --repetitions=5 --mergefun=sign_max
```

- One modality (BL-single experiments in the paper):
```python
python ../mains/mj_trainUWYHGaitNet_DataGen_CasiaB_1mod.py --experdir=OUTPATH --infodir DATAPATH/casiab_info/  --datadir DATAPATH/CASIAB_tf/ --prefix=ugaitnet_blsingle_casia --nclasses=74 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=64 --extraepochs=25 --datatype=2 --gaitset --repetitions=8

```

- Early fusion (BL-all experiments in the paper):
```python
python ../mains/mj_trainUWYHGaitNet_DataGen_CasiaB.py --dbbasedir=DATAPATH/CASIAB_tf/ --experdir=OUTPATH --infodir DATAPATH/casiab_info/  --datadir DATAPATH/CASIAB_tf/ --prefix=ugaitnet_blall_casia --nclasses=74 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=gray --bs=40 --extraepochs=25 --datatype=2 --gaitset --repetitions=5 --mergefun=sign_max --nomissing
```

#### 2. Testing in an open world scenario
```python
python ../mains/mj_testUWYHGaitNet_open_casiab.py  --datadir DATAPATH/CASIAB_tf/tfimdb_casia_b_N050_test_nm05-06_000_of25_60x60 --datadirtrain DATAPATH/tfimdb_casia_b_N050_ft_of25_60x60 --usemod1=1 --usemod2=1 --modality gray --modality0 of --verbose 1 --useavg=1 --bs=64 --knn=3 --usemirror 0 --typecode=3 --gaitset --nclasses 50 --model MODELPATH/model-state-0074.hdf5 --nametype 2
```

Comments:
* `allcombostest`: evaluates all possible combinations of missing modalities. If not used, you have to properly set `usemodX` arguments.
* `typecode`: 1 for longest gait signature, 2 for shortest gait signature (e.g. 256 dims).
* `usemirror`: 1 for adding mirror samples during the evaluation.
* `useavg`: use average to combine subsequences?
* `usemod1`: use the first modality?
* `usemod2`: use the second modality?

**Warning**: to save time, gallery codes are saved to disk if not found. But, be careful, as you might be using old or incorrect ones if you change models between evalutations.

### Both Datasets (TUM-GAID + CASIA-B)
#### 1. Training
- Missing modalities:
```python
python mains/mj_trainUWYHGaitNet_DataGen_2mod_BothDatasets.py --dbbasedir=DATAPATH/both_datasets/ --experdir=OUTPATH --infodir DATAPATH/casiab_info/  --datadir DATAPATH/both_datasets/ --prefix=ugaitnet_both --nclasses=224 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=of+gray --bs=18 --extraepochs=25 --datatype=2 --use3d --gaitset
```

- One modality (BL-single experiments in the paper):
```python
python ../mains/mj_trainUWYHGaitNet_DataGen_1mod_BothDatasets.py --experdir=OUTPATH --infodir DATAPATH/casiab_info/  --datadir DATAPATH/both_datasets/ --prefix=ugaitnet_both_blsingle --nclasses=224 --epochs=75 --lr=0.0001 --dropout=0.4 --ndense=0 --margin=0.2  --optimizer=Adam --casenet=D --wid=0.1 --wver=1.0 --mod=of --bs=40 --extraepochs=25 --datatype=2 --use3d --gaitset
```

#### 2. Testing in an open world scenario
Just use the previous test codes with the model obtained from the training on both datasets.

## References
M. Marín-Jiménez, F. Castro, R. Delgado-Escaño, V. Kalogeiton,  N. Guil. _"UGaitNet: Multimodal gait recognition with missing input modalities"_. IEEE TIFS, 2021

