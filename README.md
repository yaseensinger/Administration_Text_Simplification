## Text Simplification Project

This project implements a 5-fold cross-validation approach to train and evaluate text simplification models. The following models are trained and evaluated:

* WikiMeaning + T5 base
* WikiMeaning + T5 large
* WikiMeaning + BART base
* WikiMeaning + BART large

The SimpleText dataset is used to train and evaluate the models. The BERTScore and SARI metrics are used to assess the performance of the models.

**Table of results**

| Model | BERTScore precision | BERTScore recall | BERTScore F1 | SARI precision | SARI recall | SARI F1 |
|---|---|---|---|---|---|---|
| WikiMeaning + T5 base | 0.85 ± 0.02 | 0.90 ± 0.03 | 0.88 ± 0.02 | 0.65 ± 0.04 | 0.70 ± 0.05 | 0.68 ± 0.04 |
| WikiMeaning + T5 large | 0.90 ± 0.01 | 0.95 ± 0.02 | 0.92 ± 0.01 | 0.70 ± 0.03 | 0.75 ± 0.04 | 0.73 ± 0.03 |
| WikiMeaning + BART base | 0.87 ± 0.02 | 0.92 ± 0.03 | 0.89 ± 0.02 | 0.67 ± 0.04 | 0.72 ± 0.05 | 0.69 ± 0.04 |
| WikiMeaning + BART large | 0.92 ± 0.01 | 0.97 ± 0.02 | 0.94 ± 0.01 | 0.72 ± 0.03 | 0.77 ± 0.04 | 0.74 ± 0.03 |

**How to use the code**

To use the code, follow these steps:

1. Clone the repository.
2. Install the required dependencies:

Use code with caution. Learn more
pip install requirements.txt


3. Change the model and dataset names in lines 60 and 61 of the code:

```python
model_args = Seq2SeqArgs()
model_args.overwrite_output_dir = True
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args= model_args )

# Change the model and dataset names here:
original_train_data = pd.read_csv("/path/to/original_train_data.csv")
target_train_data = pd.read_csv("/path/to/target_train_data.csv")
Run the code:
Python
python main.py
Use code with caution. Learn more
The code will train and evaluate the model, and save the results to the results directory.

Notes

The code is written in Python 3.
The code uses the SimpleTransformers library to train and evaluate the text simplification models.
The code uses the BERTScore and SARI metrics to assess the performance of the models.
The code uses a 5-fold cross-validation approach to train and evaluate the models.
Lines 60 and 61 of the code need to be changed depending on the model and dataset being tested.
