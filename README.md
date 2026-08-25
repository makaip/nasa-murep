# Unsupervised SD-CDOM detection on the Yucatan Peninsula with Isolation Forests

A cleaned-up codebase for an upcoming paper: *The detection of sediment derived CDOM using remote sensing in the Gulf of Mexico*. We use isolation forests (IFs) on MODIS-derrived remote sensing features to classify upwelling regions. Specifically, we use sea surface temperature (SST) and sea surface level (SSL) data as our input features.

### File Structure Layout

- `data/`
  - `pipelines/` custom L2 and L3 MODIS processing scripts
  - `preprocessing/` pickling nessecary features
- `experiments/`
  - `isolation_forest/` IF model training & testing
  - `visualization/` for figure making
    - `histograms/` plotting weekly histograms of raw features
    - `raw/` plotting monthly maps of raw features
    - `violins/` plotting monthly comparisons of features inside and outside classified upwelling regions
