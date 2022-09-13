# -*- coding: utf-8 -*-
# @Time    : 2021/2/11 10:19
# @Author  : miliang
# @FileName: train_utils.py
# @Software: PyCharm

from models.bert_base import Model
# from models.bert_crf import Model
from tqdm import tqdm
from optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import copy
import datetime


def train(config, train_iter, dev_iter):
    model = Model(config).to(config.device)
    model.train()


    bert_param_optimizer = list(model.pre_model.named_parameters())
    linear_param_optimizer = list(model.hidden2label.named_parameters())
    # crf_param_optimizer = list(model.crf_layer.named_parameters())


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # pre-train model
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "lr":config.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":config.bert_learning_rate },
        # linear layer
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,"lr": config.bert_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,"lr": config.bert_learning_rate},

        # crf,单独设置学习率
        # {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "lr": config.crf_learning_rate},
        # {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr": config.crf_learning_rate}

    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(len(train_iter) * config.epoch * config.warmup_prop),
                                                num_training_steps=len(train_iter) * config.epoch)

    # optimizer = BertAdam(optimizer_grouped_parameters, lr=config.bert_learning_rate, warmup=config.warmup_prop, schedule="warmup_cosine",
    #                      t_total=len(train_iter) * config.epoch, max_grad_norm=config.clip_grad)

    cum_step = 0
    for i in range(config.epoch):
        model.train()
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists in tqdm(train_iter, position=0, desc='训练中'):
            loss, _ = model.forward(input_ids_list, input_mask_list, segment_ids_list, label_ids_list)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            # loss.backward()
            # optimizer.step()
            # model.zero_grad()
            cum_step += 1

        P, R, F = set_test(config, model, dev_iter)
        config.logger.info('dev set :  epoch_{}, step_{},precision_{:.4f},recall_{:.4f},F1_{:.4f}'.format(i, cum_step, P, R, F))


def get_save_path(config):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def set_test(config, model, dev_iter):
    y_pred_label_list, y_true_label_list = [], []
    model.eval()
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists in tqdm(dev_iter, position=0, desc='测试中'):
            _, predict = model.forward(input_ids_list, input_mask_list, segment_ids_list, label_ids_list)
            # predict = predict.data.cpu().numpy()

            temp_predict_list = make_lables(config, predict, char_lists)
            temp_true_list = make_lables(config, label_ids_list, char_lists)

            y_true_label_list.extend(temp_true_list)
            y_pred_label_list.extend(temp_predict_list)

    f1, P, R = get_entity_evaluate(config, y_pred_label_list, y_true_label_list)
    return P, R, f1

def get_entity_evaluate(config, y_pred_label_list, y_ture_label_list):
    TP_list, true_num_list, pre_num_list = [0] * len(config.entity_label), [0] * len(config.entity_label), [0] * len(config.entity_label)
    for index, items in enumerate(y_pred_label_list):
        predict_dict = y_pred_label_list[index]
        ture_dict = y_ture_label_list[index]
        for label_index, label_type in enumerate(config.entity_label):
            predict_list = [str(i["start"]) + "_" + str(i["end"]) + "_" + str(i["content"]) for i in predict_dict[label_type]]
            ture_list = [str(i["start"]) + "_" + str(i["end"]) + "_" + str(i["content"]) for i in ture_dict[label_type]]

            ture_list_change = copy.deepcopy(ture_list)
            for y_pred in predict_list:
                if y_pred in ture_list_change:
                    ture_list_change.remove(y_pred)
                    TP_list[label_index] += 1
            true_num_list[label_index] += len(ture_list)
            pre_num_list[label_index] += len(predict_list)
    P_list, R_list, f1_list = get_single_entity(TP_list, true_num_list, pre_num_list)
    for index, _ in enumerate(config.entity_label):
        config.logger.info(config.entity_label[index], "-->true_num:{},pre_num:{},TP_num:{},P:{:.4f},R:{:.4f},f1:{:.4f}".format(
            true_num_list[index], pre_num_list[index], TP_list[index], P_list[index], R_list[index], f1_list[index]))
    P, R, f1 = get_total_entity(TP_list, true_num_list, pre_num_list)
    return f1, P, R

def get_single_entity(TP_list, true_num_list, pre_num_list):
    P_list, R_list, f1_list = [0]*len(TP_list), [0]*len(TP_list), [0]*len(TP_list)
    for index, _ in enumerate(TP_list):
        P_list[index] = TP_list[index] / pre_num_list[index] if pre_num_list[index] != 0 else 0
        R_list[index] = TP_list[index] / true_num_list[index] if true_num_list[index] != 0 else 0
        f1_list[index] = 2 * P_list[index] * R_list[index] / (P_list[index] + R_list[index]) if (P_list[index] + R_list[index])!=0 else 0
    return P_list, R_list, f1_list

def get_total_entity(TP_list, true_num_list, pre_num_list):
    TP = sum(TP_list)
    true_num = sum(true_num_list)
    pre_num = sum(pre_num_list)
    P = TP / pre_num if pre_num != 0 else 0
    R = TP / true_num if pre_num != 0 else 0

    f1 = 2 * P * R / (P + R) if (P + R)!=0 else 0
    return P, R, f1


def make_lables(config, predict, char_lists):
    batch_bio_list = []
    for index, _ in enumerate(predict):
        temp_dict = make_label(config, predict[index], char_lists[index])
        batch_bio_list.append(temp_dict)
    return batch_bio_list

def make_label(config, predict, char_lists):
    label_dict = {label_type: [] for label_type in config.entity_label}
    for char_index, _ in enumerate(predict):
        if predict[char_index] >= 4 and predict[char_index] % 2 == 0:
            type_index = predict[char_index]
            start_index = char_index
            end_index = start_index + 1
            if end_index >= len(predict):
                break
            if predict[end_index] == type_index + 1:
                while predict[end_index] == type_index + 1:
                    end_index += 1
                    if len(predict) <= end_index:
                        break
                end_index -= 1
                temp_dict = {"start": start_index - 1, "end": end_index,
                             "content": "".join(char_lists[start_index: end_index + 1])}
                label_dict[config.entity_label[(type_index - 4) // 2]].append(temp_dict)

    return label_dict

