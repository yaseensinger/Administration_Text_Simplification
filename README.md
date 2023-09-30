## Text Simplification Project

This project implements a 5-fold cross-validation approach to train and evaluate text simplification models and data sets 

The following datasets are trained and evaluated:
* WikiAuto
* NewsAuto
* Simpa
* Wiki + Simpa
* NewsAuto + Simpa
  
The following models are trained and evaluated:
* T5 base
* T5 large
* Bart Base
* bart large
  
The SimpleText dataset is used to train and evaluate the models. The BERTScore and SARI metrics are used to assess the performance of the models.

**Results**

The results of the text simplification experiments are available in the following Excel file:

https://github.com/yaseensinger/Administration_Text_Simplification/blob/main/NLP_data.xlsx

You can use the Excel file to compare the performance of different models and to identify the best model for your specific needs.

**How to use the code**

To use the code, follow these steps:

1. Clone the repository.
2. Install the required dependencies:

Use code with caution. Learn more
pip install requirements.txt

Change the model and dataset names in lines 60 and 61 of the code, depending on the model and dataset you want to train and evaluate:
```python
model_args = Seq2SeqArgs()
model_args.overwrite_output_dir = True
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args= model_args )
```
4. Run the code:

```python
python main.py
```

The code will train and evaluate the model, and save the results to the `results` directory.

**Notes**

* The code is written in Python 3.
* The code uses the SimpleTransformers library to train and evaluate the text simplification models.
* The code uses the BERTScore and SARI metrics to assess the performance of the models.
* The code uses a 5-fold cross-validation approach to train and evaluate the models.
* Lines 60 and 61 of the code need to be changed depending on the model and dataset being tested.
