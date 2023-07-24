from delete_object import InpaintingModel
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
import os,base64 
import time
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_ubyte

import sys
sys.path.append("/data/mmdetection/")
from mmdet.apis import inference_detector, init_detector 

def jpg_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

#Object Detection Model
config_file = '/data/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
# trained model
checkpoint_file = '/data/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
detectmodel = init_detector(config_file, checkpoint_file, device='cuda:0')
filepath = "/data/snli_ve_test.tsv"

uniq_id_old_list = []
image_id_old_list = []
caption_old_list = []
hypothesis_old_list = []
imagebase64_old_list = []
label_old_list = []
final_list =[]
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split('\t')  
        uniq_id_old_list.append(line[0])
        image_id_old_list.append(line[1])
        imagebase64_old_list.append(line[2])
        hypothesis_old_list.append(line[3])
        caption_old_list.append(line[4])
        label_old_list.append(line[5])

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou

def MR1(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    for index,each in enumerate(newform_list):
        flag = 0
        bbox_delete_list = each["bbox"].split(",")
        bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
        rest_list = newform_list[:index] + newform_list[index+1:]
        for item in rest_list:
            item_bboxlist = item["bbox"].split(",")
            item_bbox = [float(item_bboxlist[0]),float(item_bboxlist[1]),float(item_bboxlist[2]),float(item_bboxlist[3])]
            if calculate_iou(item_bbox, bbox_delete) > 0.05:
                flag = 1
                break
        if flag == 1:
            continue
        object_delete = each["object"]
        deleted_sentence_list = []
        #Reorganize the remaining semantic subunits
        for num, rest in enumerate(rest_list):
            sub_sentence = ""
            if object_delete in rest["property"]:
                deleted_sentence_list.append(rest["object"])
                continue
            else:
                if rest["object"] == "":
                    object_index = -1
                else:
                    first_object = rest["object"].split()[0]
                    object_index = rest["sentence"].find(first_object)
                if rest["property"] == "":
                    property_index = -1
                else:
                    first_property = rest["property"].split()[0]
                    property_index = rest["sentence"].find(first_property)
                if object_index < property_index:
                    deleted_sentence_list.append((rest["object"] + " " + rest["property"]).strip())
                else:
                    deleted_sentence_list.append((rest["property"]+ " " + rest["object"]).strip())
        delete_sentence =  " and ".join(deleted_sentence_list)
        new_deletesentence_list.append(delete_sentence)
        bbox = (float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3]))
        input_path = jpg_path
        output_path = "/data/lama-main/output/snli_MR1/" + uniq_id + "_" + str(index) + ".jpg" 
        pic_base64  = jpg_to_base64(output_path)
        new_imagebase64_list.append(pic_base64)

    return new_imagebase64_list,new_deletesentence_list
    


def MR2(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    for index,each in enumerate(newform_list):
        flag = 0
        bbox_delete_list = each["bbox"].split(",")
        bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
        if len(newform_list) > 1:
            rest_list = newform_list[:index] + newform_list[index+1:]
            for item in rest_list:
                item_bboxlist = item["bbox"].split(",")
                item_bbox = [float(item_bboxlist[0]),float(item_bboxlist[1]),float(item_bboxlist[2]),float(item_bboxlist[3])]
                if calculate_iou(item_bbox, bbox_delete) > 0.2:
                    flag = 1
                    break
            if flag == 1:
                continue
        bbox = (float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3]))
        output_path = "/data/lama-main/output/esnli_MR2/" + uniq_id + "_" + str(index) + ".jpg" 

        pic_base64  = jpg_to_base64(output_path)
        new_imagebase64_list.append(pic_base64)
    return new_imagebase64_list,new_deletesentence_list

def MR3(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    classes = detectmodel.dataset_meta["classes"]
    result = inference_detector(detectmodel, jpg_path)
    preds = result.pred_instances
    bboxes = preds.bboxes
    scores = preds.scores
    labels = preds.labels
    detectbox_list = []
    for num, each in enumerate(bboxes):
        if scores[num] > 0.3:
            tempbox = [float(each[0]),float(each[1]),float(each[2]),float(each[3])]
            detectbox_list.append(tempbox)
    for newnum,box in enumerate(detectbox_list):
        flag = 0
        for index,each in enumerate(newform_list):
            bbox_delete_list = each["bbox"].split(",")
            bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
            if calculate_iou(box, bbox_delete) > 0.1:
                flag = 1
                break
        if flag == 1:
            continue
        else:
            input_path = jpg_path
            output_path = "/data/lama-main/output/esnli_MR2_2/" + uniq_id + "_" + str(newnum) + ".jpg" 
            pic_base64  = jpg_to_base64(output_path)
            new_imagebase64_list.append(pic_base64)          
    return new_imagebase64_list,new_deletesentence_list

object_path = "/data/OFA-main/data/cv_data/snli_ve/result/snli_ve_test_bbox_entailment.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("fail in reading {} ".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
uniq_list = [] 
hypothesis_list = []
subject_list = []
modifier_list = []
label_list = []
sentence_list = []
bbox_list = []
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
    bbox = row_list[6:7]
    uniq_list.append(uniq_id[0])
    subject_list.append(subject[0])
    modifier_list.append(modifier[0])
    label_list.append(label[0])
    sentence_list.append(sentence[0])
    bbox_list.append(bbox[0])
newform_list = []
for num,each in enumerate(subject_list):
    templist = []
    object = subject_list[num]
    property = modifier_list[num]
    bbox = bbox_list[num]
    objectnew_list = object.split(";")
    propertynew_list = property.split(";")
    bboxnew_list =  bbox.split(";")
    origin_sentence = sentence_list[num]

    for num_two,each in enumerate(objectnew_list):
        if num_two == 0:
            continue
        if objectnew_list[num_two] == "" and propertynew_list[num_two] == "":
            continue
        try:
            temp_dic = {}
            temp_dic["object"] = objectnew_list[num_two]
            temp_dic["property"] = propertynew_list[num_two]
            temp_dic["bbox"] = bboxnew_list[num_two]
            temp_dic["sentence"] = origin_sentence
            templist.append(temp_dic)
        except IndexError:
            continue
    newform_list.append(templist)
imagebase64_alllist = []
deletesentence_alllist = []
uniqid_all_list = []
for num,each in enumerate(newform_list):
    print(num)
    jpg_path = "/data/OFA-main/data/cv_data/snli_ve/testimage/" + uniq_list[num] + ".jpg"
    if len(each) > 1:
        new_imagebase64_list,new_deletesentence_list = MR1(each,jpg_path,uniq_list[num])
        imagebase64_alllist.extend(new_imagebase64_list)
        deletesentence_alllist.extend(new_deletesentence_list)
        tempuniqid_list = [uniq_list[num]]*len(new_deletesentence_list)
        uniqid_all_list.extend(tempuniqid_list)

out_path = "/data/OFA-main/data/cv_data/snli_ve/result/snli_test_MR1.tsv"
with open(out_path, 'w', newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    for newnum, each in enumerate(deletesentence_alllist):
        try:
            index = uniq_id_old_list.index(uniqid_all_list[newnum])
            templine = []
            templine.append(uniqid_all_list[newnum] + "_" + str(newnum))
            templine.append(image_id_old_list[index])
            templine.append(imagebase64_alllist[newnum])
            templine.append(each)
            templine.append(caption_old_list[index])
            templine.append(label_old_list[index])
            tsv_w.writerow(templine) 
        except ValueError:
            print(newnum)
            continue




