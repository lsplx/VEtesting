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

        object_path = "D:/snli_test_entity.xlsx"
        try:
            pd_sheets = pd.ExcelFile(object_path)
        except Exception as e:
            print("read {} file fail".format(object_path), e)
        df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
        final_list = []
        for row in df.itertuples(index=True):
            templist = []
            row_list = list(row)
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
        df1.to_excel('D:/snli_test_entity.xlsx', sheet_name='Sheet1', index=False)        
    return predictions

object_path = "D:/snli_ve_test.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("read {} file fail".format(object_path), e)
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
predictions = extract_entities(uniq_list,hypothesis_list,label_list)


