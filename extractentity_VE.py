import json
from tkinter import E
import Levenshtein
from pandas import DataFrame
import openai
import time
import pandas as pd
from openpyxl import load_workbook
def edit_distance_score(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    similarity_score = 1 - (distance / max_length)
    return similarity_score

openai.api_key = "your_apikey"

def compute_metrics_subject(predictions, labels):
    tp = fp = fn = 0
    fp_flag = 0
    for prediction, label in zip(predictions, labels):
        for each in label:
            if edit_distance_score(prediction.strip(),each) > 0.8:
                tp += 1
                fp_flag = 1
            else:
                fn += 1
    if fp_flag == 0:
        fp += 1
        # if prediction and label:  # True Positive
        #     tp += 1
        # elif prediction and not label:  # False Positive
        #     fp += 1
        # elif not prediction and label:  # False Negative
        #     fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def compute_metrics_modifier(predictions, labels):
    tp = fp = fn = 0
    for prediction, label in zip(predictions, labels):
        for each in prediction:
            fp_flag = 0
            for every in label:
                if  edit_distance_score(each.strip(),every) > 0.6:
                    tp += 1
                    fp_flag = 1
            if fp_flag == 0:
                fp += 1
        for each in label:
            fn_flag = 0
            for every in prediction:    
                if edit_distance_score(each,every.strip()) > 0.6:
                    fn_flag = 1
            if fn_flag == 0:
                fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def evaluate_model(predictions, test_data):
    subjects_predictions = [p['subject'] for p in predictions]
    subjects_labels = []
    for data in test_data:
        templist = []
        for entity in data["entities"]:
            if  entity['type'] == '主体':
                templist.append(entity['entity'].strip())
        subjects_labels.append(templist)
    modifiers_predictions = [p['modifiers'] for p in predictions]
    modifiers_labels = []
    for data in test_data:
        templist = []
        for entity in data["entities"]:
            if  entity['type'] == '特征':
                templist.append(entity['entity'].strip())
        modifiers_labels.append(templist)

    subject_precision, subject_recall, subject_f1 = compute_metrics_subject(subjects_predictions, subjects_labels)
    modifier_precision, modifier_recall, modifier_f1 = compute_metrics_modifier(modifiers_predictions, modifiers_labels)

    final_list = []
    for num,each in enumerate(subjects_predictions) :
        templist = []
        combined_string = ' '.join(test_data[num]['tokens'])
        templist.append(combined_string)
        templist.append(each)
        subjectstring_gth = ""
        for subject in subjects_labels[num]:
            subjectstring_gth =subjectstring_gth + "; " +  subject 
        templist.append(subjectstring_gth)
        modifierstring = ""
        for modifier in modifiers_predictions[num]:
            modifierstring = modifierstring+ "; " + modifier  
        templist.append(modifierstring)
        modifierstring_gth = ""
        for modifier_gth in modifiers_labels[num]:
            modifierstring_gth = modifierstring_gth + "; " + modifier_gth 
        templist.append(modifierstring_gth)
        final_list.append(templist)
    # 创建一个新的 Excel 工作簿
    df = DataFrame(final_list,columns=["sentence","sub_predict","sub_gth","modifier_predict","modifier_gth"])
    df.to_excel('D:/test400_predictgpt_nocotext.xlsx', sheet_name='Sheet1', index=False)


    print("Subject Precision:", subject_precision)
    print("Subject Recall:", subject_recall)
    print("Subject F1:", subject_f1)
    print("Modifier Precision:", modifier_precision)
    print("Modifier Recall:", modifier_recall)
    print("Modifier F1:", modifier_f1)


def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    return test_data

def extract_entities(uniq_list,hypothesis_list,label_list):
    predictions = []
    for num, item in enumerate(hypothesis_list):
        print(num)
        sentence = item
        if num < 14726:
            continue
        if sentence == "a":
            continue
        if label_list[num] != "entailment":
            continue

        prompt = '''
        Please extract the subject and modifier in the following sentence, the sentence is divided into multiple semantic sub-units, each sub-unit contains the subject and modifier, and output it in json format and in English:
        Let me give you some example:
        example1:
        Input:the boy is outdoors.
        Output:
        [
        {
             "subject": "boy" ,
             "modifier": "is outdoors"
        }  
        ]

        example2:
        Input:the worker fixes the window.
        Output:
        [
        {
             "subject": "worker" ,
             "modifier": "fixes the window"
        },
        {
            "subject": "window" ,
             "modifier": ""
        }
        ]

        example3:
        Input:a woman is near a kid.
        Output:
        [
        {
             "subject": "woman" ,
             "modifier": "is near a kid"
        },
        {
             "subject": "kid" ,
             "modifier": ""
        }  
        ]

        example4:
        Input:There is a man and a boy flying a kite in the park
        Output:
        [
        {
             "subject": "man" ,
             "modifier": "flying a kite in the park"
        },
        {
             "subject": "boy" ,
             "modifier": "flying a kite in the park"
        },  
        {
             "subject": "kite" ,
             "modifier": ""
        }
        ]

        example5:
        Input:A woman in a dress walks down a driveway.
        Output:
        [
        {
             "subject": "woman" ,
             "modifier": "in a dress walks down a driveway"
        }
        ]

        example6:
        Input:A boy is running on the beach while girls watch
        Output:
        [
        {
             "subject": "boy" ,
             "modifier": "is running on the beach"
        },
        {
             "subject": "girls" ,
             "modifier": "watch"
        }
        ]

        example7:
        Input:a guy and girl working on a bicycle
        Output:
        [
        {
             "subject": "guy" ,
             "modifier": "working on a bicycle"
        },
        {
             "subject": "girl" ,
             "modifier": "working on a bicycle"
        },
        {
             "subject": "bicycle" ,
             "modifier": ""
        }
        ]
        
        Input: 
        ''' + sentence + "\nOutput:"
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.RateLimitError :
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.APIError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.ServiceUnavailableError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        except openai.error.APIConnectionError:
            time.sleep(80)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
        output = completion.choices[0].message["content"]
        try:
            output_json = json.loads(output_string)
        except json.decoder.JSONDecodeError:
            index = output_string.find('}')
            if index != -1:
                output_string = output_string[:index+1] + ',' + output_string[index+1:]
            output_json = json.loads(output_string)
        final_list = []
        templist = []
        subject_new = ""
        modifier_new = ""
        for each in output_json:
            subject_new = subject_new + ";" +  each["subject"]
            modifier_new = modifier_new + ";" + each["modifier"]
        # templist.append(uniq_list[num])
        # templist.append(subject)
        # templist.append(modifier)
        # templist.append(label_list[num])
        # final_list.append(templist)

        object_path = "D:/esnli_test_entity.xlsx"
        try:
            pd_sheets = pd.ExcelFile(object_path)
        except Exception as e:
            print("读取 {} 文件失败".format(object_path), e)
        df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
        # uniq_list = [] 
        # subject_list = []
        # modifier_list = []
        # label_list = []
        # num = 0
        final_list = []
        for row in df.itertuples(index=True):
            templist = []
            row_list = list(row)
            #uniq_id
            uniq_id = row_list[1:2]
            subject = row_list[2:3]
            modifier = row_list[3:4]
            label = row_list[4:5]
            sentence_old = row_list[5:6]
            templist.append(uniq_id[0])
            templist.append(subject[0])
            templist.append(modifier[0])
            templist.append(label[0])
            templist.append(sentence_old[0])
            final_list.append(templist)
        templist = []
        templist.append(uniq_list[num])
        templist.append(subject_new)
        templist.append(modifier_new)
        templist.append(label_list[num])
        templist.append(sentence)
        final_list.append(templist)
        df1 = DataFrame(final_list,columns=["uniq_id","subject","modifier","label","sentence"])
        df1.to_excel('D:/esnli_test_entity.xlsx', sheet_name='Sheet1', index=False)        
    return predictions




object_path = "D:/esnli_ve_test.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("读取 {} 文件失败".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
uniq_list = [] 
hypothesis_list = []
caption_list = []
label_list = []
num = 0
final_list = []
for row in df.itertuples(index=True):
    row_list = list(row)

    uniq_id = row_list[1:2]
    hypothesis = row_list[2:3]
    caption = row_list[3:4]
    label = row_list[4:5]
    uniq_list.append(uniq_id[0])
    hypothesis_list.append(hypothesis[0])
    label_list.append(label[0])

# test_data = [{"tokens": ["the", "red", "luggage"], "entities": [{"entity": "luggage ", "start": 2, "end": 3, "type": "subject", "newid": 0, "id": 0}, {"entity": "red ", "start": 1, "end": 2, "type": "feature", "newid": 1, "id": 1}], "relations": []}]
predictions = extract_entities(uniq_list,hypothesis_list,label_list)


