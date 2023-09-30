import sys
import numpy as np
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from easse.sari import get_corpus_sari_operation_scores
from easse.bertscore import corpus_bertscore
from easse.sari import corpus_sari
from datasets import load_dataset
# Read the text file
with open("/content/drive/MyDrive/project/original.txt", "r", encoding="utf-8") as f:
    original_line = f.readlines()
    
with open("/content/drive/MyDrive/project/target.txt", "r", encoding="utf-8") as f:
    target_lines = f.readlines()

if len(original_line)!=len(target_lines):
  print("files are not of the same length")


# Divide the original_line into 5 separate chunks
chunk_size = len(original_line) // 5
original_chunks = [original_line[i:i+chunk_size] for i in range(0, len(original_line), chunk_size)]
target_chunks = [target_lines[i:i+chunk_size] for i in range(0, len(target_lines), chunk_size)]


add_list = [] 
keep_list = []
del_list = []
sari_list= []

bert_p_list =[]
bert_R_list = []
bert_F1_list =[]


for i in range(5):
  
    # Create first split(test data)
    original_test_data = pd.DataFrame(original_chunks[i], columns=["input_text"])
    target_test_data = pd.DataFrame(target_chunks[i], columns=["target_text"])
    
    # Create second split with all other chunks concatenated for target data(train data)
    target_other_chunks = [target_chunks[j] for j in range(5) if j != i]
    traget_train_data = pd.concat([pd.DataFrame(chunk, columns=["target_text"]) for chunk in target_other_chunks], ignore_index=True)

    # Create second split with all other chunks concatenated for original data (train data)
    origianl_other_chunks = [original_chunks[j] for j in range(5) if j != i]
    original_train_data = pd.concat([pd.DataFrame(chunk, columns=["input_text"]) for chunk in origianl_other_chunks], ignore_index=True)


    # create data frame 
    train_df = pd.concat([original_train_data, traget_train_data], axis=1)
    
    # Use trained Seq2Seq model to predict simplified lines
    # train the model using original_train_data and traget_train_data
    
    model_args = Seq2SeqArgs()
    model_args.overwrite_output_dir = True
    model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args= model_args )
    model.train_model(train_df)

    predictions = model.predict(list(original_test_data["input_text"]))
    print(predictions)
    print(original_test_data)
    
    #strip data of /n from both test a and train data 
    target_test_list = list(target_test_data["target_text"])
    target_test_list  = [line.strip() for line in  target_test_list ]

    original_test_list = list(original_test_data["input_text"])
    original_test_list  = [line.strip() for line in  original_test_list ]
    
    #calculate the sari socres (add,keep and delete) and add them to togerther each loop
    sari  = get_corpus_sari_operation_scores(orig_sents= original_test_list, sys_sents= predictions, refs_sents=[target_test_list])
    bert = corpus_bertscore(sys_sents= predictions, refs_sents=[target_test_list])
    sari_score = corpus_sari(orig_sents= original_test_list, sys_sents= predictions, refs_sents=[target_test_list])

    bert_p_list.append(bert[0])
    bert_R_list.append(bert[1])
    bert_F1_list.append(bert[2])

    sari_list.append(sari_score)
    add_list.append (sari[0])
    keep_list.append (sari[1])
    del_list.append (sari[2])
    

#calcuate the average 
bert_p_average = sum(bert_p_list)/5
bert_R_average = sum(bert_R_list)/5
bert_F1_average = sum(bert_F1_list)/5

add_average = sum(add_list)/5
keep_average = sum(keep_list)/5
del_average = sum(del_list)/5
sari_average = sum(sari_list)/5
#calculate the standard deviation 


bert_p_std = np.std(bert_p_list)
bert_R_std = np.std(bert_R_list)
bert_F1_std = np.std(bert_F1_list)


add_std = np.std(add_list)
keep_std = np.std(keep_list)
del_std = np.std(del_list)
sari_std = np.std(sari_list)

print("bertscore precision average: ", bert_p_average)
print("bertscore Recall average: ", bert_R_average)
print("bertscore F1 average: ", bert_F1_average)
print("\n")

print("add average: ", add_average,)
print("keep average: ", keep_average)
print("delete average: ", del_average)
print("sari average: ", sari_average)
print("\n")

print("bertscore precision standard deviation: ", bert_p_std,)
print("bertscore recall standard deviation: ", bert_R_std,)
print("bertscore F2 standard deviation: ", bert_F1_std,)
print("\n")

print("add standard deviation: ", add_std,)
print("keep standard deviation: ", keep_std)
print("delete standard deviation: ", del_std)
print("Sari standard deviation: ", sari_std)

