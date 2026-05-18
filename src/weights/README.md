
# Model weights

Download the pretrained feature extractor checkpoint from [here](https://unilj-my.sharepoint.com/:u:/g/personal/zbabnik_fe1_uni-lj_si/IQA00fgiftoeQpERSoAGcpGnAfEOQ-GaGOD1wwIpacR9u7U?e=wb82ji).

Place the downloaded file under `src/weights` and name it `feature_extractor.pth` so it matches the default inference config:

```bash
mkdir -p src/weights
mv /path/to/downloaded_checkpoint.pth src/weights/feature_extractor.pth