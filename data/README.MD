# Fukuchi2017 Dataset

The data in this folder comes from a [publicly available dataset](http://dx.doi.org/10.6084/m9.figshare.4543435) released with the paper:

**Fukuchi RK, Fukuchi CA & Duarte M (2017). A public dataset of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics. *Peer J*, 5: e3298. doi: [10.7717/peerj.3298](https://doi.org/10.7717/peerj.3298)**

Any additional use must be in line with that described in the original paper.

## Steps for Extracting Data

1. Download the .zip file from the link above or provided in the paper.
2. Extract the archive and place the files in a folder named `raw` <u>**in the same folder as this README**</u>.
3. Run the **`extractData.py`** script. This will take the necessary files out from the raw folder and sort them appropriately. It will also convert the `.c3d` files to necessary OpenSim formats (i.e. `.trc`, `.mot`, `.xml`).
4. After running this script, the `raw` folder with the original files can be deleted.

The above process results in separate folders for each participant from the dataset (i.e. RBDS*n* where *n* is the participant ID number). Within these folders are all of the extracted and create files relevant to that participant. Note that only a sub-sample (i.e. from RBDS001 through to RBDS028) of the original dataset are extracted for this work.

## Corrected Data Issues

- RBDS001 runT25 R.MT1 and R.MT5 markers swapped around. Corrected these issues in `.trc` file after converting.
- RBDS03 runT25 R.MT1 marker originally labelled as R.MT5, and R.MT5 marker originally labelled as R.ASY. Corrected these issues in `.trc` file after converting.