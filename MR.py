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
sys.path.append("/data/czy/mmdetection/")
from mmdet.apis import inference_detector, init_detector 

def jpg_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

#目标检测模型
config_file = '/data/czy/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
# 训练好的模型
checkpoint_file = '/data/czy/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
detectmodel = init_detector(config_file, checkpoint_file, device='cuda:0')

# 创建模型实例
model = InpaintingModel("pretrained/fine_tuned/states_pt_places2.pth")
filepath = "/data/czy/OFA-main/data/cv_data/snli_ve/snli_ve_test.tsv"
# filepath = "/data/czy/OFA-main/data/cv_data/snli_ve/result/esnli_ve/esnli_ve_test.tsv"
uniq_id_old_list = []
image_id_old_list = []
caption_old_list = []
hypothesis_old_list = []
imagebase64_old_list = []
label_old_list = []
final_list =[]
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        # 读取一行后，末尾一般会有一个\n，所以用strip函数去掉
        line = line.strip('\n').split('\t')  
        uniq_id_old_list.append(line[0])
        image_id_old_list.append(line[1])
        imagebase64_old_list.append(line[2])
        hypothesis_old_list.append(line[3])
        caption_old_list.append(line[4])
        label_old_list.append(line[5])

def calculate_iou(bbox1, bbox2):
    # 计算两个边界框的相交部分的坐标
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # 计算相交部分的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算两个边界框的面积
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # 计算 IOU 值
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou

def MR1(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    for index,each in enumerate(newform_list):
        #设置判断删除区域是否和剩余区域有过度重叠的标志
        flag = 0
        bbox_delete_list = each["bbox"].split(",")
        bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
        #构建删除子单元后的剩余但与组
        rest_list = newform_list[:index] + newform_list[index+1:]
        #先判断要删除的区域是否会影响剩余的区域
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
        #对剩余的语义子单元进行重组
        for num, rest in enumerate(rest_list):
            #首先判断删除的object是否在剩余子单元的property中
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
        #删除图片中的文本对应区域
        bbox = (float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3]))
        # 调用模型进行批量处理
        input_path = jpg_path
        output_path = "/data/czy/lama-main/output/snli_MR1/" + uniq_id + "_" + str(index) + ".jpg" 
        # output_path = "/data/czy/OFA-main/data/cv_data/snli_ve/testimage_delete/" + uniq_id + "_" + str(index) + ".jpg" 
        # model.inpainting(input_path, bbox, output_path)
        # print(f"Saved output file at: {output_path}")
        #判断修复的图像和原图像的相似度
        image_origin = Image.open(input_path)
        image_delete = Image.open(output_path)
        width, height = max(image_origin.width, image_delete.width), max(image_origin.height, image_delete.height)
        image_origin = image_origin.resize((width, height))
        image_delete = image_delete.resize((width, height))
        original_image = img_as_ubyte(image_origin.convert('RGB'))
        repaired_image = img_as_ubyte(image_delete.convert('RGB'))
        ssim_score = ssim(original_image, repaired_image,win_size=7, channel_axis=2)
        if ssim_score > 0.05:
            #转为base64
            pic_base64  = jpg_to_base64(output_path)
            new_imagebase64_list.append(pic_base64)
        else:
            #用简单的白色遮蔽
            image = Image.open(jpg_path)
            mask_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            # 获取屏蔽区域的左上角和右下角坐标
            left, top, right, bottom = bbox
            # 在屏蔽图像上绘制白色矩形，覆盖屏蔽区域
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255))
            # 将屏蔽图像叠加到原始图像上
            masked_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
            rgb_im = masked_image.convert('RGB')
            # rgb_im.save('audacious.jpg')
            # 保存屏蔽后的图像
            output_path_another = "/data/czy/OFA-main/data/cv_data/snli_ve/testimage_delete/" + uniq_id + "_" + str(index) + "white.jpg" 
            rgb_im.save(output_path_another)
            pic_base64  = jpg_to_base64(output_path_another)
            new_imagebase64_list.append(pic_base64)

            # if each["bbox"]
    return new_imagebase64_list,new_deletesentence_list
    


def MR2(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    for index,each in enumerate(newform_list):
        #设置判断删除区域是否和剩余区域有过度重叠的标志
        flag = 0
        bbox_delete_list = each["bbox"].split(",")
        bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
        #判断Object的个数
        if len(newform_list) > 1:
        #构建删除子单元后的剩余但与组
            rest_list = newform_list[:index] + newform_list[index+1:]
            #先判断要删除的区域是否会影响剩余的区域
            for item in rest_list:
                item_bboxlist = item["bbox"].split(",")
                item_bbox = [float(item_bboxlist[0]),float(item_bboxlist[1]),float(item_bboxlist[2]),float(item_bboxlist[3])]
                if calculate_iou(item_bbox, bbox_delete) > 0.2:
                    flag = 1
                    break
            if flag == 1:
                continue
        #删除图片中的文本对应区域
        bbox = (float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3]))
        # 调用模型进行批量处理
        input_path = jpg_path
        output_path = "/data/czy/lama-main/output/esnli_MR2/" + uniq_id + "_" + str(index) + ".jpg" 
        # model.inpainting(input_path, bbox, output_path)
        # print(f"Saved output file at: {output_path}")
        #判断修复的图像和原图像的相似度
        image_origin = Image.open(input_path)
        image_delete = Image.open(output_path)
        width, height = max(image_origin.width, image_delete.width), max(image_origin.height, image_delete.height)
        image_origin = image_origin.resize((width, height))
        image_delete = image_delete.resize((width, height))
        original_image = img_as_ubyte(image_origin.convert('RGB'))
        repaired_image = img_as_ubyte(image_delete.convert('RGB'))
        ssim_score = ssim(original_image, repaired_image,win_size=7, channel_axis=2)
        if ssim_score > 0.05:
            pic_base64  = jpg_to_base64(output_path)
            new_imagebase64_list.append(pic_base64)
        else:
            #用简单的白色遮蔽
            image = Image.open(jpg_path)
            mask_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            # 获取屏蔽区域的左上角和右下角坐标
            left, top, right, bottom = bbox
            # 在屏蔽图像上绘制白色矩形，覆盖屏蔽区域
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255))
            # 将屏蔽图像叠加到原始图像上
            masked_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
            rgb_im = masked_image.convert('RGB')
            # rgb_im.save('audacious.jpg')
            # 保存屏蔽后的图像
            output_path_another = "/data/czy/OFA-main/data/cv_data/snli_ve/testimage_delete/" + uniq_id + "_" + str(index) + "white.jpg" 
            rgb_im.save(output_path_another)
            pic_base64  = jpg_to_base64(output_path_another)
            new_imagebase64_list.append(pic_base64)

            # if each["bbox"]
    return new_imagebase64_list,new_deletesentence_list

def MR2_2(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    #先检测出图像中所有的区域信息
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
            # x, y, w, h = each[0], each[1], each[2], each[3]
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((x, y, w, h), width=2, outline='#41fc59')
            # font = ImageFont.truetype("/data/czy/mmdetection/Arial.ttf", 15, encoding="unic")
            # label = classes[labels[num]]
            # draw.text((x,y), label, "red", font=font)
    for newnum,box in enumerate(detectbox_list):
        flag = 0
        for index,each in enumerate(newform_list):
            #判断检测区域是否和文本对应区域有重合
            bbox_delete_list = each["bbox"].split(",")
            bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
            if calculate_iou(box, bbox_delete) > 0.1:
                flag = 1
                break
        if flag == 1:
            continue
        else:
             # 调用模型进行批量处理
            input_path = jpg_path
            output_path = "/data/czy/lama-main/output/esnli_MR2_2/" + uniq_id + "_" + str(newnum) + ".jpg" 
            # model.inpainting(input_path, box, output_path)
            # print(f"Saved output file at: {output_path}")
            #判断修复的图像和原图像的相似度
            image_origin = Image.open(input_path)
            image_delete = Image.open(output_path)
            width, height = max(image_origin.width, image_delete.width), max(image_origin.height, image_delete.height)
            image_origin = image_origin.resize((width, height))
            image_delete = image_delete.resize((width, height))
            original_image = img_as_ubyte(image_origin.convert('RGB'))
            repaired_image = img_as_ubyte(image_delete.convert('RGB'))
            ssim_score = ssim(original_image, repaired_image,win_size=7, channel_axis=2)
            if ssim_score > 0.05:
                pic_base64  = jpg_to_base64(output_path)
                new_imagebase64_list.append(pic_base64)
            else:
                #用简单的白色遮蔽
                image = Image.open(jpg_path)
                mask_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
                # 获取屏蔽区域的左上角和右下角坐标
                left, top, right, bottom = box
                # 在屏蔽图像上绘制白色矩形，覆盖屏蔽区域
                draw = ImageDraw.Draw(mask_image)
                draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255))
                # 将屏蔽图像叠加到原始图像上
                masked_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
                rgb_im = masked_image.convert('RGB')
                # rgb_im.save('audacious.jpg')
                # 保存屏蔽后的图像
                output_path_another = "/data/czy/OFA-main/data/cv_data/snli_ve/image_MR2_2/" + uniq_id + "_" + str(newnum) + "white.jpg" 
                rgb_im.save(output_path_another)
                pic_base64  = jpg_to_base64(output_path_another)
                new_imagebase64_list.append(pic_base64)
    return new_imagebase64_list,new_deletesentence_list

def MR3(newform_list,jpg_path,uniq_id):
    new_imagebase64_list = []
    new_deletesentence_list = []
    for index,each in enumerate(newform_list):
        #设置判断删除区域是否和剩余区域有过度重叠的标志
        flag = 0
        bbox_delete_list = each["bbox"].split(",")
        bbox_delete = [float(bbox_delete_list[0]),float(bbox_delete_list[1]),float(bbox_delete_list[2]),float(bbox_delete_list[3])]
        #构建删除子单元后的剩余但与组
        rest_list = newform_list[:index] + newform_list[index+1:]
        #先判断要删除的区域是否会影响剩余的区域
        for item in rest_list:
            item_bboxlist = item["bbox"].split(",")
            item_bbox = [float(item_bboxlist[0]),float(item_bboxlist[1]),float(item_bboxlist[2]),float(item_bboxlist[3])]
            if calculate_iou(item_bbox, bbox_delete) > 0.2:
                flag = 1
                break
        if flag == 1:
            continue
        object_delete = each["object"]
        deleted_sentence_list = []
        #对剩余的语义子单元进行重组
        for num, rest in enumerate(rest_list):
            #首先判断删除的object是否在剩余子单元的property中
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

            # if each["bbox"]
    return new_imagebase64_list,new_deletesentence_list



#读取bbox以及subsentence
object_path = "/data/czy/OFA-main/data/cv_data/snli_ve/result/snli_ve_test_bbox_entailment.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("读取 {} 文件失败".format(object_path), e)
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
    # subentence_list = []
    #转变形式[{O:,P:,B,S},...]
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
    jpg_path = "/data/czy/OFA-main/data/cv_data/snli_ve/testimage/" + uniq_list[num] + ".jpg"
    # jpg_path = "/data/czy/flickr 30k/flickr30k-images/" + uniq_list[num].split("#")[0]
    #MR2不需要规定2个以上的Object
    if len(each) > 1:
        new_imagebase64_list,new_deletesentence_list = MR1(each,jpg_path,uniq_list[num])
        imagebase64_alllist.extend(new_imagebase64_list)
        deletesentence_alllist.extend(new_deletesentence_list)
        tempuniqid_list = [uniq_list[num]]*len(new_deletesentence_list)
        uniqid_all_list.extend(tempuniqid_list)
print("NUM")
print(len(uniqid_all_list))

out_path = "/data/czy/OFA-main/data/cv_data/snli_ve/result/snli_test_entailment_MR1new.tsv"
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

#生成原始蕴含数据集
# out_path = "/data/czy/OFA-main/data/cv_data/snli_ve/result/snli_test_entailment_nohypo.tsv"
# with open(out_path, 'w', newline='') as f:
#     tsv_w = csv.writer(f, delimiter='\t')
#     for newnum, each in enumerate(uniq_id_old_list):
#         if label_old_list[newnum].strip() == "entailment":
#             templine = []
#             templine.append(uniq_id_old_list[newnum] )
#             templine.append(image_id_old_list[newnum])
#             templine.append(imagebase64_old_list[newnum])
#             templine.append("")
#             templine.append(caption_old_list[newnum])
#             templine.append(label_old_list[newnum])
#             tsv_w.writerow(templine) 


