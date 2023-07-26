import cv2
import numpy as np
import random
# from imagecorruptions import corrupt
import csv
import os,base64 
from pandas import DataFrame
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from PIL import Image, ImageDraw
import pandas as pd


object_path = "/data/esnli_test_entity.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("read fail in {}".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
uniq_list = [] 
hypothesis_list = []
subject_list = []
modifier_list = []
label_list = []
sentence_list = []
num = 0
final_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    #uniq_id
    uniq_id = row_list[1:2]
    subject = row_list[2:3]
    modifier = row_list[3:4]
    label = row_list[4:5]
    sentence = row_list[5:6]
    uniq_list.append(uniq_id[0])
    subject_list.append(subject[0])
    modifier_list.append(modifier[0])
    label_list.append(label[0])
    sentence_list.append(sentence[0])

filepath = "/data/esnli_ve_test.tsv"
uniqall_id_list = []
imageall_id_list = []
captionall_list = []
hypothesisall_list = []
imagebase64_list = []
labelall_list = []
final_list =[]
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split('\t')  
        uniqall_id_list.append(line[0])
        imageall_id_list.append(line[1])
        imagebase64_list.append(line[2])
        hypothesisall_list.append(line[3])
        captionall_list.append(line[4])
        labelall_list.append(line[5])
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_en')

bbox_list = []
for num ,uniq_id in enumerate(uniq_list):
    print(num)
    # create object-property sentence
    object = subject_list[num]
    property = modifier_list[num]
    object_list = object.split(";")
    property_list = property.split(";")
    origin_sentence = sentence_list[num]
    subentence_list = []
    for num_two,each in enumerate(object_list):
        if num_two == 0:
            continue
        if object_list[num_two] == "":
            object_index = -1
        else:
            first_object = each.split()[0]
            object_index = origin_sentence.find(first_object)
        if property_list[num_two] == "":
            property_index = -1
        else:
            first_property = property_list[num_two].split()[0]
            property_index = origin_sentence.find(first_property)
        if object_index == -1 and property_index == -1:
            continue
        if object_index < property_index:
            subentence_list.append((each + " " + property_list[num_two]).strip())
        else:
            subentence_list.append((property_list[num_two] + " " + each).strip())
    bbox_all = ""
    for num_three, subsentence in enumerate(subentence_list):
        jpg_path = "/data/flickr 30k/flickr30k-images/" + uniq_id.split("#")[0]
        image = Image.open(jpg_path)
        input = {'image': jpg_path, 'text': subsentence}
        draw = ImageDraw.Draw(image)
        bbox = ofa_pipe(input)[OutputKeys.BOXES][0] 
        x1, y1, x2, y2 = bbox
        tempbbox = str(x1) + "," +str(y1) + "," + str(x2) + "," + str(y2)
        bbox_all = bbox_all + ";" + tempbbox

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)


        output_path = "/data/esnli_test_subVG/" + uniq_id + "_" + str(num_three) + ".jpg"  # 替换为你要保存的图像路径
        image.save(output_path)
        # print(num)
    bbox_list.append(bbox_all)
final_list = []
for num,each in enumerate(modifier_list):
    templist = []
    templist.append(uniq_list[num])
    templist.append(subject_list[num])
    templist.append(modifier_list[num])
    templist.append(label_list[num])
    templist.append(sentence_list[num])
    templist.append(bbox_list[num])
    final_list.append(templist)
df = DataFrame(final_list,columns=["uniq_id","subject","modifier","label","sentence","bbox"])
df.to_excel('/data/esnli_ve_test_bbox_entailment.xlsx', sheet_name='Sheet1', index=False)
