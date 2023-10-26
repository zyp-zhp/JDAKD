import argparse
import random
from sampler import data_sampler
from config import Config
import torch
from model.bert_encoder import Bert_Encoder
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer, Proto_Softmax_Layer
from data_loader import get_data_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import collections
from copy import deepcopy
from collections import Counter
import pickle
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            labels, _, tokens = batch_data
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            maxsize=max([x.to(config.device).size(0) for x in tokens])
            padded=[F.pad(x.to(config.device),(0,maxsize-x.to(config.device).size(0))) for x in tokens]
            tokens = torch.stack(padded,dim=0)
            # tokens = torch.stack([x.to(config.device) for x in tokens if x.size()==tokens[0].size()],dim=0)
            # tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            #print(tokens.shape)
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def contrastive_loss(hidden, labels):

    logsoftmax = nn.LogSoftmax(dim=-1)

    return -(logsoftmax(hidden) * labels).sum() / labels.sum()


def construct_hard_triplets(output, labels, relation_data):
    positive = []
    negative = []
    pdist = nn.PairwiseDistance(p=2)
    for rep, label in zip(output, labels):
        positive_relation_data = relation_data[label.item()]
        negative_relation_data = []
        for key in relation_data.keys():
            if key != label.item():
                negative_relation_data.extend(relation_data[key])
        positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
        negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
        positive_index = torch.argmax(positive_distance)
        negative_index = torch.argmin(negative_distance)
        positive.append(positive_relation_data[positive_index.item()])
        negative.append(negative_relation_data[negative_index.item()])


    return positive, negative


def train_first(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, _, tokens) in enumerate(data_loader):

            optimizer.zero_grad()

            logits_all = []
            maxsize=max([x.to(config.device).size(0) for x in tokens])
            padded=[F.pad(x.to(config.device),(0,maxsize-x.to(config.device).size(0))) for x in tokens]
            tokens = torch.stack(padded,dim=0)
            # tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps = encoder(tokens)
            outputs,_ = dropout_layer(reps)
            positives,negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1 + loss2 + tri_loss

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")




def compute_cam_attention_maps(encoder, classifier, input_tokens):
    encoder.eval()  # Set the encoder to evaluation mode
    classifier.eval()  # Set the classifier to evaluation mode

    with torch.no_grad():
        feature_maps = encoder(input_tokens)
        attention_weights = classifier(feature_maps)

    encoder.train()  # Set the encoder back to training mode
    classifier.train()  # Set the classifier back to training mode

    return attention_weights


def compute_cam_distillation_loss(attention_maps_base, attention_maps_fine_tuned):
    # Calculate the difference between attention maps
    difference = attention_maps_base - attention_maps_fine_tuned

    # Compute a loss based on the difference (e.g., MSE or other similarity loss)
    distillation_loss = torch.mean(torch.square(difference))

    return distillation_loss



def train_mem_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data,
                prev_encoder, prev_dropout_layer, prev_classifier,prev_encoder1, prev_dropout_layer1, prev_classifier1, prev_relation_index,prev_relation_index1):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    distill_criterion = nn.CosineEmbeddingLoss()
    T = config.kl_temp
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, _, tokens) in enumerate(data_loader):

            optimizer.zero_grad()

            logits_all = []
            maxsize=max([x.to(config.device).size(0) for x in tokens])
            padded=[F.pad(x.to(config.device),(0,maxsize-x.to(config.device).size(0))) for x in tokens]
            tokens = torch.stack(padded,dim=0)
            #tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]

            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps = encoder(tokens)
            normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            outputs,_ = dropout_layer(reps)


            if prev_dropout_layer is not None:
                prev_outputs, _ = prev_dropout_layer(reps)
                positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1 + loss2 + tri_loss


            if prev_encoder is not None and prev_encoder1 is not None:
                prev_reps = prev_encoder(tokens).detach()
                prev_reps1 = prev_encoder1(tokens).detach()

                normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)
                normalized_prev_reps_emb1 = F.normalize(prev_reps1.view(-1, prev_reps1.size()[1]), p=2, dim=1)
                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                feature_distill_loss1 = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb1,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                # loss += feature_distill_loss
                # loss += feature_distill_loss1
                loss =loss+0.5*feature_distill_loss+0.5*feature_distill_loss1



            if prev_dropout_layer is not None and prev_classifier is not None and prev_dropout_layer1 is not None and prev_classifier1 is not None:
                prediction_distill_loss = None
                dropout_output_all = []
                prev_dropout_output_all = []
                # prev_dropout_output_all1 = []
                for i in range(config.f_pass):
                    output, _ = dropout_layer(reps)
                    prev_output, _ = prev_dropout_layer(reps)
                    # prev_output1, _ = prev_dropout_layer1(reps)

                    dropout_output_all.append(output)
                    prev_dropout_output_all.append(output)
                    # prev_dropout_output_all1.append(output)
                    # prev_relation_index1=prev_relation_index1.to(torch.int64)

                    pre_logits = prev_classifier(output).detach()
                    # pre_logits1 = prev_classifier1(output).detach()

                    pre_logits = F.softmax(pre_logits.index_select(1, prev_relation_index) / T, dim=1)
                    # pre_logits1 = F.softmax(pre_logits1.index_select(1, prev_relation_index1) / T, dim=1)


                    log_logits = F.log_softmax(logits_all[i].index_select(1, prev_relation_index) / T, dim=1)
                    # log_logits1 = F.log_softmax(logits_all[i].index_select(1, prev_relation_index1) / T, dim=1)
                    if i == 0:
                        prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        # prediction_distill_loss1= -torch.mean(torch.sum(pre_logits1 * log_logits1, dim=1))

                    else:
                        prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        # prediction_distill_loss1 += -torch.mean(torch.sum(pre_logits1 * log_logits1, dim=1))

                prediction_distill_loss /= config.f_pass
                # prediction_distill_loss1 /= config.f_pass

                loss += prediction_distill_loss
                # loss = loss+0.5*prediction_distill_loss+0.5*prediction_distill_loss1


                dropout_output_all = torch.stack(dropout_output_all)
                prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))



                # prev_dropout_output_all1 = torch.stack(prev_dropout_output_all1)
                #
                # mean_prev_dropout_output_all1 = torch.mean(prev_dropout_output_all1, dim=0)
                #
                # normalized_prev_output1 = F.normalize(
                #     mean_prev_dropout_output_all1.view(-1, mean_prev_dropout_output_all1.size()[1]), p=2, dim=1)
                # hidden_distill_loss1 = distill_criterion(normalized_output, normalized_prev_output1,
                #                                         torch.ones(tokens.size(0)).to(
                #                                             config.device))
                loss += hidden_distill_loss
                # loss += hidden_distill_loss1
                # loss=loss+0.5*hidden_distill_loss+0.5*hidden_distill_loss1


            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")



def batch2device(batch_tuple, device):
    ans = []
    for var in batch_tuple:
        if isinstance(var, torch.Tensor):
            ans.append(var.to(device))
        elif isinstance(var, list):
            ans.append(batch2device(var))
        elif isinstance(var, tuple):
            ans.append(tuple(batch2device(var)))
        else:
            ans.append(var)
    return ans


def evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data, seen_relations, map_relid2tempid):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)

        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n
# def evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data, seen_relations, map_relid2tempid):
#     data_loader = get_data_loader(config, test_data, batch_size=1)
#     encoder.eval()
#     dropout_layer.eval()
#     classifier.eval()
#     n = len(test_data)
#
#     correct = 0
#     class_correct = {}
#     class_total = {}
#
#     for step, batch_data in enumerate(data_loader):
#         labels, _, tokens = batch_data
#         labels = labels.to(config.device)
#         labels = [map_relid2tempid[x.item()] for x in labels]
#         labels = torch.tensor(labels).to(config.device)
#
#         tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
#         reps = encoder(tokens)
#         reps, _ = dropout_layer(reps)
#         logits = classifier(reps)
#
#         seen_relation_ids = [rel2id[relation] for relation in seen_relations]
#         seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
#         seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
#         max_smi = np.max(seen_sim,axis=1)
#
#         label_smi = logits[:,labels].cpu().data.numpy()
#
#         if label_smi >= max_smi:
#             correct += 1
#
#         for i, label in enumerate(labels):
#             label_id = label.item()
#             if label_id not in class_correct:
#                 class_correct[label_id] = 0
#                 class_total[label_id] = 0
#             class_total[label_id] += 1
#             if label_smi[i] >= max_smi[i]:
#                 class_correct[label_id] += 1
#
#     accuracy = correct / n
#     class_accuracy = {label_id: class_correct[label_id] / class_total[label_id] for label_id in class_correct}
#
#     return accuracy, class_accuracy


def select_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1].cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(relation_dataset))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory


def get_proto(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1]
        features.append(feature)
    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True).cpu()
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    return proto, standard


def generate_relation_data(protos, relation_standard):
    relation_data = {}
    relation_sample_nums = 10
    for id in protos.keys():
        relation_data[id] = []
        difference = np.random.normal(loc=0, scale=1, size=relation_sample_nums)
        for diff in difference:
            relation_data[id].append(protos[id] + diff * relation_standard[id])
    return relation_data


def generate_current_relation_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    relation_data = []
    encoder.eval()
    dropout_layer.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1].cpu()
        relation_data.append(feature)
    return relation_data

from transformers import  BertTokenizer
def data_augmentation(config, encoder, train_data, prev_train_data):
    expanded_train_data = train_data[:]
    expanded_prev_train_data = prev_train_data[:]
    encoder.eval()
    all_data = train_data + prev_train_data
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    entity_index = []
    entity_mention = []
    for sample in all_data:
        e11 = sample['tokens'].index(30522)
        e12 = sample['tokens'].index(30523)
        e21 = sample['tokens'].index(30524)
        e22 = sample['tokens'].index(30525)
        entity_index.append([e11,e12])
        entity_mention.append(sample['tokens'][e11+1:e12])
        entity_index.append([e21,e22])
        entity_mention.append(sample['tokens'][e21+1:e22])

    data_loader = get_data_loader(config, all_data, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        feature1, feature2 = torch.split(feature, [config.encoder_output_size,config.encoder_output_size], dim=1)
        features.append(feature1)
        features.append(feature2)
    features = torch.cat(features, dim=0)
    # similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    similarity_matrix = []
    for i in range(len(features)):
        similarity_matrix.append([0]*len(features))

    for i in range(len(features)):
        for j in range(i,len(features)):
            similarity = F.cosine_similarity(features[i],features[j],dim=0)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    similarity_matrix = torch.tensor(similarity_matrix).to(config.device)
    zero = torch.zeros_like(similarity_matrix).to(config.device)
    diag = torch.diag_embed(torch.diag(similarity_matrix))
    similarity_matrix -= diag
    similarity_matrix = torch.where(similarity_matrix<0.95, zero, similarity_matrix)
    nonzero_index = torch.nonzero(similarity_matrix)
    expanded_train_count = 0

    for origin, replace in nonzero_index:
        sample_index = int(origin/2)
        sample = all_data[sample_index]
        if entity_mention[origin] == entity_mention[replace]:
            continue
        new_tokens = sample['tokens'][:entity_index[origin][0]+1] + entity_mention[replace] + sample['tokens'][entity_index[origin][1]:]
        if len(new_tokens) < config.max_length:
            new_tokens = new_tokens + [0]*(config.max_length-len(new_tokens))
        else:
            new_tokens = new_tokens[:config.max_length]

        new_sample = {
            'relation': sample['relation'],
            'neg_labels': sample['neg_labels'],
            'tokens': new_tokens
        }
        if sample_index < len(train_data) and expanded_train_count < 5 * len(train_data):
            expanded_train_data.append(new_sample)
            expanded_train_count += 1
        else:
            expanded_prev_train_data.append(new_sample)
    return expanded_train_data, expanded_prev_train_data

def data_augmentation_aca(config, encoder, train_data, prev_train_data):
    expanded_train_data = train_data[:]
    expanded_prev_train_data = prev_train_data[:]
    encoder.eval()
    # 这两行代码将编码器设置为评估模式，并将训练数据和先前训练数据合并为一个列表。
    all_data = train_data + prev_train_data
    # 使用BERT模型的tokenizer创建一个tokenizer对象，用于对文本进行分词。
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    entity_index = []
    entity_mention = []
    # 它遍历所有的数据样本，找到特定的标记（30522、30523、30524、30525）的索引，
    # 并将实体的索引和实体的提及添加到相应的列表中。
    for sample in all_data:
        e11 = sample['tokens'].index(30522)
        e12 = sample['tokens'].index(30523)
        e21 = sample['tokens'].index(30524)
        e22 = sample['tokens'].index(30525)
        entity_index.append([e11,e12])
        entity_mention.append(sample['tokens'][e11+1:e12])
        entity_index.append([e21,e22])
        entity_mention.append(sample['tokens'][e21+1:e22])

    data_loader = get_data_loader(config, all_data, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        # 这段代码将特征张量拆分为两个部分，并将它们添加到特征列表中
        feature1, feature2 = torch.split(feature, [config.encoder_output_size,config.encoder_output_size], dim=1)
        features.append(feature1)
        features.append(feature2)
    # 这行代码将特征列表中的特征张量连接起来，形成一个新的特征张量。
    features = torch.cat(features, dim=0)
    # similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

    # 这段代码创建了一个相似度矩阵，用于存储特征之间的余弦相似度。
    # 它遍历特征张量中的每对特征，并计算它们之间的余弦相似度，并将结果存储在相似度矩阵中。
    similarity_matrix = []
    for i in range(len(features)):
        similarity_matrix.append([0]*len(features))

    for i in range(len(features)):
        for j in range(i,len(features)):
            similarity = F.cosine_similarity(features[i],features[j],dim=0)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    # 这段代码将相似度矩阵转换为张量，并进行一些操作。
    # 首先，创建一个全零张量zero，与相似度矩阵具有相同形状和设备。
    # 然后，创建一个对角矩阵diag，其中对角线元素为相似度矩阵的对角线元素。
    # 接下来，将相似度矩阵减去对角矩阵，将对角线元素置零。
    # 最后，将相似度矩阵中小于0.95的元素置零。
    similarity_matrix = torch.tensor(similarity_matrix).to(config.device)
    zero = torch.zeros_like(similarity_matrix).to(config.device)
    diag = torch.diag_embed(torch.diag(similarity_matrix))
    similarity_matrix -= diag
    similarity_matrix = torch.where(similarity_matrix<0.95, zero, similarity_matrix)
    nonzero_index = torch.nonzero(similarity_matrix)
    expanded_train_count = 0

    # 这段代码遍历相似度矩阵中非零元素的索引，并根据一些条件生成新的样本。
    # 对于每个非零元素，它获取原始样本和替换样本的索引，并根据这些索引生成新的令牌序列。
    # 然后，根据新的令牌序列创建一个新的样本，并将其添加到扩展后的训练数据或扩展后的先前训练数据中。
    for origin, replace in nonzero_index:
        sample_index = int(origin/2)
        sample = all_data[sample_index]
        if entity_mention[origin] == entity_mention[replace]:
            continue
        new_tokens1 = sample['tokens'][:entity_index[origin][0]+1] + entity_mention[replace] + sample['tokens'][entity_index[origin][1]:]
        new_tokens2 = sample['tokens'][:entity_index[replace][0]+1] + entity_mention[origin] + sample['tokens'][entity_index[replace][1]:]
        new_tokens=[101]+new_tokens1+new_tokens2+[102]
        if len(new_tokens) < config.max_length:
            new_tokens = new_tokens + [0]*(config.max_length-len(new_tokens))
        else:
            new_tokens = new_tokens[:config.max_length]
        new_sample = {
            'relation': sample['relation'],
            'neg_labels': sample['neg_labels'],
            'tokens': new_tokens
        }
        if sample_index < len(train_data) and expanded_train_count < 5 * len(train_data):
            expanded_train_data.append(new_sample)
            expanded_train_count += 1
        else:
            expanded_prev_train_data.append(new_sample)

    return expanded_train_data, expanded_prev_train_data

def calculate_relation_weights_data_distribution(training_data):
    relation_counts=Counter(training_data)
    total_relations=len(training_data)

    relation_weights = {relation:count / total_relations for relation, count in relation_counts.items()}
    return relation_weights

def get_relation_embeddings(config, encoder, dropout_layer, classifier, relation_dataset,map_relid2tempid):
    data_loader = get_data_loader(config, relation_dataset, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    embeddings = []
    lables_list=[]

    for step, batch_data in enumerate(data_loader):

        labels, _, tokens = batch_data
        lables_list.extend(labels.cpu().numpy())
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            embeddings.append(reps.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)


    return embeddings,lables_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tacred", type=str)
    parser.add_argument("--shot", default=10, type=str)
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    config.task = args.task
    config.shot = args.shot
    config.step1_epochs = 5
    config.step2_epochs = 15
    config.step3_epochs = 20
    config.temperature =0.08

    if config.task == "FewRel":
        config.relation_file = "data/fewrel/relation_name.txt"
        config.rel_index = "data/fewrel/rel_index.npy"
        config.rel_feature = "data/fewrel/rel_feature.npy"
        config.rel_des_file = "data/fewrel/relation_description.txt"
        config.num_of_relation = 80
        if config.shot == 5:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_5/test_0.txt"
        elif config.shot == 10:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
        else:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
    else:
        config.relation_file = "data/tacred/relation_name.txt"
        config.rel_index = "data/tacred/rel_index.npy"
        config.rel_feature = "data/tacred/rel_feature.npy"
        config.num_of_relation = 41
        if config.shot == 5:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_5/test_0.txt"
        else:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_10/test_0.txt"

    result_cur_test = []
    result_whole_test = []
    bwt_whole = []
    fwt_whole = []
    X = []
    Y = []
    relation_divides = []
    for i in range(10):
        relation_divides.append([])
    for rou in range(config.total_round):
        test_cur = []
        test_total = []
        random.seed(config.seed+rou*100)
        sampler = data_sampler(config=config, seed=config.seed+rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        id2sentence = sampler.get_id2sent()
        encoder = Bert_Encoder(config=config).cuda()
        dropout_layer = Dropout_Layer(config=config).to(config.device)
        num_class = len(sampler.id2rel)

        memorized_samples = {}
        memory = collections.defaultdict(list)
        history_relations = []
        history_data = []
        prev_relations = []
        classifier = None
        prev_classifier = None
        prev_encoder = None
        prev_dropout_layer = None
        prev_classifier1 = None
        prev_encoder1 = None
        prev_dropout_layer1 = None
        prev_relation_index1 = []

        relation_standard = {}
        forward_accs = []
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print(current_relations)

            prev_relations = history_relations[:]
            train_data_for_initial = []
            count = 0
            for relation in current_relations:
                history_relations.append(relation)
                train_data_for_initial += training_data[relation]
                relation_divides[count].append(float(rel2id[relation]))
                count += 1


            temp_rel2id = [rel2id[x] for x in seen_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            prev_relation_index = []

            # embeddings,lable = get_relation_embeddings(config, encoder, dropout_layer, classifier, train_data_for_initial,
            #                                          map_relid2tempid)
            # # np.savetxt("beforerelation.txt", beforerelation)
            # with open(f'beforerelation{rou}_{steps}.pkl','wb') as f:
            #     pickle.dump({'embeddings':embeddings,'lables':lable},f)

            prev_samples = []
            for relation in prev_relations:
                prev_relation_index.append(map_relid2tempid[rel2id[relation]])

                prev_samples += memorized_samples[relation]
            prev_relation_index = torch.tensor(prev_relation_index).to(config.device)
            # if(steps==0):
            #     for relation in prev_relations:
            #         prev_relation_index1.append(map_relid2tempid[rel2id[relation]])
            #     prev_relation_index1 = torch.tensor(prev_relation_index1).to(config.device)



            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(history_relations)).to(
                config.device)

            print('开始增强')
            torch.cuda.empty_cache()
            if config.aca:
                aca_train_data_for_initial,aca_expanded_prev_samples = data_augmentation_aca(config, encoder, train_data_for_initial, prev_samples)
            temp_mem={}
            temp_protos = {}
            # relation_weights = calculate_relation_weights_data_distribution(current_relations)
            for relation in current_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, training_data[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_protos[rel2id[relation]] = proto * relation_weights[relation]
            # relation_weights = calculate_relation_weights_data_distribution(prev_relations)
            for relation in prev_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_protos[rel2id[relation]] = proto * relation_weights[relation]

            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            if steps != 0:
                forward_acc = evaluate_strict_model(config, prev_encoder, prev_dropout_layer, classifier, test_data_1, seen_relations, map_relid2tempid)
                forward_accs.append(forward_acc)

            # train_simple_model(config, encoder, dropout_layer, classifier, aca_train_data_for_initial, config.step1_epochs, map_relid2tempid)
            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step1_epochs, map_relid2tempid)

            print(f"simple finished")


            temp_protos = {}
            # relation_weights = calculate_relation_weights_data_distribution(current_relations)
            for relation in current_relations:
                proto, standard = get_proto(config,encoder,dropout_layer,training_data[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_protos[rel2id[relation]] = proto * relation_weights[relation]
                relation_standard[rel2id[relation]] = standard

            # relation_weights = calculate_relation_weights_data_distribution(prev_relations)
            for relation in prev_relations:
                proto, _ = get_proto(config,encoder,dropout_layer,memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_protos[rel2id[relation]] = proto * relation_weights[relation]
            generator = Generator(config.hidden_size, 768)  # 输入大小为input_size，输出大小为output_size
            discriminator = Discriminator(768)  # 输入大小为input_size

            # # 定义优化器和损失函数
            # generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
            # discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
            # loss_function = nn.BCELoss()
            #
            # # 训练生成对抗网络
            # num_epochs = 100
            # for epoch in range(num_epochs):
            #     for id in temp_protos.keys():
            #         # 训练生成器
            #         generator.train()
            #         discriminator.eval()
            #         generator_optimizer.zero_grad()
            #
            #         proto = temp_protos[id]
            #
            #         # 生成新的关系数据
            #         generated_data = generator(proto)
            #
            #         # 判别器判断生成的数据的真假
            #         discriminator_output = discriminator(generated_data)
            #         generator_loss = loss_function(discriminator_output, torch.ones_like(discriminator_output))
            #
            #         # 反向传播和优化生成器
            #         generator_loss.backward()
            #         generator_optimizer.step()
            #
            #         # 训练判别器
            #         generator.eval()
            #         discriminator.train()
            #         discriminator_optimizer.zero_grad()
            #
            #         # 判别器判断真实数据的真假
            #        # real_data = torch.cat([torch.tensor(training_data[relation]) for relation in current_relations], dim=0)
            #         real_output = discriminator(proto)
            #         real_loss = loss_function(real_output, torch.ones_like(real_output))
            #
            #         # 判别器判断生成的数据的真假
            #         generated_output = discriminator(generated_data.detach())
            #         generated_loss = loss_function(generated_output, torch.zeros_like(generated_output))
            #
            #         # 反向传播和优化判别器
            #         discriminator_loss = real_loss + generated_loss
            #         discriminator_loss.backward()
            #         discriminator_optimizer.step()
            # new_relation_data = {}
            # # 使用生成器生成新的关系数据
            # for id in temp_protos.keys():
            #     proto=temp_protos[id]
            #     new_relation_data[id]=[]
            #     new_relation_data[id].append(generator(proto))

            new_relation_data = {}
            new_relation_data = generate_relation_data(temp_protos, relation_standard)

            for relation in current_relations:
                new_relation_data[rel2id[relation]].extend(generate_current_relation_data(config, encoder,dropout_layer,training_data[relation]))

            expanded_train_data_for_initial, expanded_prev_samples = data_augmentation(config, encoder,train_data_for_initial,prev_samples)
            torch.cuda.empty_cache()
            print(len(train_data_for_initial))
            print(len(expanded_train_data_for_initial))

            # # train_first(config, encoder, dropout_layer, classifier, expanded_train_data_for_initial, config.step2_epochs, map_relid2tempid, new_relation_data)
            # train_first(config, encoder, dropout_layer, classifier, expanded_train_data_for_initial, config.step2_epochs, map_relid2tempid, new_relation_data,
            #             prev_encoder, prev_dropout_layer, prev_classifier, prev_encoder1, prev_dropout_layer1, prev_classifier1,prev_relation_index,prev_relation_index1)

            train_mem_model(config, encoder, dropout_layer, classifier, expanded_train_data_for_initial, config.step3_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier,prev_encoder1, prev_dropout_layer1, prev_classifier1, prev_relation_index,prev_relation_index1)
            print(f"first finished")

            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, dropout_layer, training_data[relation])
                memory[rel2id[relation]] = select_data(config, encoder, dropout_layer, training_data[relation])

            train_data_for_memory = []
            # train_data_for_memory += expanded_prev_samples
            train_data_for_memory += prev_samples
            for relation in current_relations:
                train_data_for_memory += memorized_samples[relation]
            print(len(seen_relations))
            print(len(train_data_for_memory))

            temp_protos = {}
            # relation_weights = calculate_relation_weights_data_distribution(seen_relations)
            for relation in seen_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto
                # temp_protos[rel2id[relation]] = proto * relation_weights[relation]

            train_mem_model(config, encoder, dropout_layer, classifier, train_data_for_memory, config.step3_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier,prev_encoder1, prev_dropout_layer1, prev_classifier1, prev_relation_index,prev_relation_index1)
            print(f"memory finished")
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            history_data.append(test_data_1)


            print(len(test_data_1))
            print(len(test_data_2))
            # cur_acc = evaluate_strict_model(config, encoder, classifier, test_data_1, seen_relations, map_relid2tempid)
            # total_acc = evaluate_strict_model(config, encoder, classifier, test_data_2, seen_relations, map_relid2tempid)

            cur_acc= evaluate_strict_model(config, encoder,dropout_layer,classifier, test_data_1, seen_relations, map_relid2tempid)
            total_acc= evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data_2, seen_relations, map_relid2tempid)

            print(f'Restart Num {rou + 1}')
            print(f'task--{steps + 1}:')
            print(f'current test acc:{cur_acc}')
            print(f'history test acc:{total_acc}')
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print(test_cur)
            print(test_total)
            # for lable_id,acc in class_acc.items():
            #     print("类别",lable_id,"acc",acc)
            # for lable_id,acc in totalclass_acc.items():
            #     print("类别",lable_id,"acc",acc)
            accuracy = []
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            for data in history_data:
                # accuracy.append(
                #     evaluate_strict_model(config, encoder, classifier, data, history_relations, map_relid2tempid))
                accuracy.append(evaluate_strict_model(config, encoder, dropout_layer, classifier, data, seen_relations, map_relid2tempid))
            print(accuracy)

            prev_encoder = deepcopy(encoder)
            prev_dropout_layer = deepcopy(dropout_layer)
            prev_classifier = deepcopy(classifier)
            if(steps==0):
                prev_encoder1 = deepcopy(encoder)
                prev_dropout_layer1 = deepcopy(dropout_layer)
                prev_classifier1 = deepcopy(classifier)

            torch.cuda.empty_cache()
            # embeddings,lable = get_relation_embeddings(config, encoder, dropout_layer, classifier, train_data_for_initial,map_relid2tempid)
            # # np.savetxt("afterrelation.txt", afterrlation)
            # with open(f'afterrlation{rou}_{steps}.pkl','wb') as f:
            #     pickle.dump({'embeddings':embeddings,'lables':lable},f)
        result_cur_test.append(np.array(test_cur))
        result_whole_test.append(np.array(test_total)*100)
        print("result_whole_test")
        print(result_whole_test)
        avg_result_cur_test = np.average(result_cur_test, 0)
        avg_result_all_test = np.average(result_whole_test, 0)
        print("avg_result_cur_test")
        print(avg_result_cur_test)
        print("avg_result_all_test")
        print(avg_result_all_test)
        std_result_all_test = np.std(result_whole_test, 0)
        print("std_result_all_test")
        print(std_result_all_test)

        accuracy = []
        temp_rel2id = [rel2id[x] for x in history_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        for data in history_data:
            accuracy.append(evaluate_strict_model(config, encoder, dropout_layer, classifier, data, history_relations, map_relid2tempid))
        print(accuracy)
        bwt = 0.0
        for k in range(len(accuracy)-1):
            bwt += accuracy[k]-test_cur[k]
        bwt /= len(accuracy)-1
        bwt_whole.append(bwt)
        fwt_whole.append(np.average(np.array(forward_accs)))
        print("bwt_whole")
        print(bwt_whole)
        print("fwt_whole")
        print(fwt_whole)
        avg_bwt = np.average(np.array(bwt_whole))
        print("avg_bwt_whole")
        print(avg_bwt)
        avg_fwt = np.average(np.array(fwt_whole))
        print("avg_fwt_whole")
        print(avg_fwt)


