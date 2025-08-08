We adopted a downsampled version of [Storm EVent ImageRy (SEVIR) dataset](https://registry.opendata.aws/sevir/) which were used in [Prediff](https://github.com/gaozhihan/PreDiff/), denoted as SEVIR-LR, where the temporal downscaling factor is 2, and the spatial downscaling factor is 3 for each dimension. 
On SEVIR-LR dataset, PreDiff generates $6\times 128\times 128$ forecasts for a given $7\times 128\times 128$ context sequence.

To download SEVIR-LR dataset directly from AWS S3, run:
```bash
cd ROOT_DIR/DDM
python ./scripts/download_sevirlr.py
```
Alternatively, if you already have the original SEVIR dataset, you may want to get SEVIR-LR by downsampling the original SEVIR. In this case run:
```bash
cd ROOT_DIR/DDM
ln -s path_to_SEVIR ./datasets/sevir  # link to your SEVIR dataset if it is not in `ROOT_DIR/DDM/datasets`
python ./scripts/downsample_sevir.py
```
