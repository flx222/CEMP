# CEMP


# 🧬 Enzyme–Substrate Model Fine-tuning and Prediction

This repository provides example scripts for **fine-tuning** and **prediction** based on the pre-trained enzyme–substrate model.  
The corresponding **dataset** and **pre-trained weights** can be accessed at Zenodo:  
👉 [https://doi.org/10.5281/zenodo.17606660](https://doi.org/10.5281/zenodo.17606660)


1. Quick Start

1. Environment Setup

   ```bash
   pip install -r requirements.txt
   ```

1. Prepare Dataset and Model

   Download the dataset and pre-trained model from Zenodo:  
   https://doi.org/10.5281/zenodo.17606660

   Organize them as:
   ```
   /path/to/dataset/
   /path/to/checkpoints/
   ```

1. Generate MindRecord Files

   ```bash
   cd example/fine-tune
   python generate_regress_smile_2x.py \
       --data_dir /path/to/dataset \
       --output_dir /path/to/output_mindrecord
   ```
   Edit `--data_dir` and `--output_dir` to match your local paths.

1. Fine-tune the Model

   ```bash
   bash quick_train.sh
   ```
   Modify the paths in `quick_train.sh` (dataset, vocab, checkpoint, output) before running.

1. Run Prediction

   ```bash
   cd example/predict
   bash quick_predict.sh
   ```
   Adjust dataset and checkpoint paths accordingly.

1. Notes

- Fine-tuning supports both regression and classification tasks.  
- Logs are saved under the specified output folder.  
- Ensure your `device_id` is correctly configured in the shell scripts.

1. Citation

If you use this repository or the associated dataset, please cite:
- Zenodo DOI: https://doi.org/10.5281/zenodo.17606660

1. Contact

For questions or collaborations, please contact:  
fanlingxitc@163.com


