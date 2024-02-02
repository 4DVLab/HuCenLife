# Segmentation.
## process
Make sure you have downloaded the dataset. 
Run the code in following:


```
python process_ground.py 
```

```
python process_hclseg.py
```
this process may be very slow, and you can also download the processed data.
And you will get the files for segmentation:
```
── Path root/
    ├──train.pkl  # info for train set
    ├──test.pkl   # info for test set
    ├──hcl_seg/
    ├──hcl_anno/
    ├──HCL_Full/
    ...
```
## train
