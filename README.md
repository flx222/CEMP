# CEMP


# 🧬 Enzyme–Substrate Model Fine-tuning and Prediction

This repository provides example scripts for **fine-tuning** and **prediction** based on the pre-trained enzyme–substrate model.  
The corresponding **dataset** and **pre-trained weights** can be accessed at Zenodo:  
👉 [https://doi.org/10.5281/zenodo.17606660](https://doi.org/10.5281/zenodo.17606660)

## 📁 Repository Structure

├── 01_generate_mr/ # Scripts for generating MindRecord files
├── 02_train_model/ # Scripts for model training
├── dataset/ # Example dataset directory
└── example/
├── fine-tune/ # Fine-tuning examples
│ ├── generate_regress_smile_2x.py # Generate MindRecord file
│ ├── quick_train.sh # Run training (edit paths)
└── predict/
├── quick_predict.sh # Run prediction (edit paths)

## 🚀 Quick Start

### 1️⃣ Environment Setup
Install dependencies (MindSpore environment recommended):

```bash
pip install -r requirements.txt

2️⃣ Prepare Dataset and Model

Download the dataset and pre-trained model from Zenodo
➡ https://doi.org/10.5281/zenodo.17606660

Then organize them as:

/path/to/dataset/
/path/to/checkpoints/

3️⃣ Generate MindRecord Files

Move to the fine-tune folder and run:

cd example/fine-tune
python generate_regress_smile_2x.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output_mindrecord

Edit --data_dir and --output_dir to match your local paths.

4️⃣ Fine-tune the Model

Execute the quick training script:

bash quick_train.sh
Modify the paths in quick_train.sh (dataset, vocab, checkpoint, output) before running.

5️⃣ Run Prediction

For prediction, go to the predict folder:
cd example/predict
bash quick_predict.sh
Adjust dataset and checkpoint paths accordingly.

🧠 Notes

Fine-tuning supports both regression and classification tasks.

Logs are automatically saved under the specified output folder.

Make sure your device_id is correctly configured in the shell scripts.

📄 Citation

If you use this repository or the associated dataset, please cite:

Zenodo DOI: 10.5281/zenodo.17606660

