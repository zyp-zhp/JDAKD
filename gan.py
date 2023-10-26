import torch
import torch.nn as nn

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






# 创建生成器和判别器模型
generator = Generator(input_size, output_size)  # 输入大小为input_size，输出大小为output_size
discriminator = Discriminator(input_size)  # 输入大小为input_size

# 定义优化器和损失函数
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
loss_function = nn.BCELoss()

# 训练生成对抗网络
num_epochs = 100
for epoch in range(num_epochs):
    # 训练生成器
    generator.train()
    discriminator.eval()
    generator_optimizer.zero_grad()

    # 生成新的关系数据
    generated_data = generator(temp_protos)

    # 判别器判断生成的数据的真假
    discriminator_output = discriminator(generated_data)
    generator_loss = loss_function(discriminator_output, torch.ones_like(discriminator_output))

    # 反向传播和优化生成器
    generator_loss.backward()
    generator_optimizer.step()

    # 训练判别器
    generator.eval()
    discriminator.train()
    discriminator_optimizer.zero_grad()

    # 判别器判断真实数据的真假
    real_data = torch.cat([training_data[relation] for relation in current_relations], dim=0)
    real_output = discriminator(real_data)
    real_loss = loss_function(real_output, torch.ones_like(real_output))

    # 判别器判断生成的数据的真假
    generated_output = discriminator(generated_data.detach())
    generated_loss = loss_function(generated_output, torch.zeros_like(generated_output))

    # 反向传播和优化判别器
    discriminator_loss = real_loss + generated_loss
    discriminator_loss.backward()
    discriminator_optimizer.step()

# 使用生成器生成新的关系数据
new_relation_data = generator(temp_protos)
#-*- coding:utf-8 -*-

def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()
    # 定义函数为交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])

    for epoch_i in range(epochs):
        #初始化一个空列表，用于储存每个批次的损失
        losses = []
        for step, batch_data in enumerate(data_loader):
            #在每个批次开始时，清零优化器的梯度
            optimizer.zero_grad()
            #从批次数据中提取标签和令牌
            labels, _, tokens = batch_data
            labels = labels.to(config.device)
            origin_labels = labels[:]
            #使用映射将标签转为临时ID
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            #将令牌堆叠成一个张量，并转移到设备上
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            #计算预测和实际标签之间的损失
            loss1 = criterion(logits, labels)
            #
            # positives, negatives = construct_hard_triplets(reps, origin_labels, training_data)
            # # 将正样本数据转移到设备上，并将其堆叠成一个张量
            # positives = torch.cat(positives, 0).to(config.device)
            # negatives = torch.cat(negatives, 0).to(config.device)
            # # 将输出数据设置为锚点
            # anchors = reps
            # # 将所有的分类结果堆叠成一个张量
            # tri_loss = triplet_loss(anchors, positives, negatives)
            # loss = loss1 + tri_loss
            # # 将损失添加到损失列表中
            # losses.append(loss.item())
            # loss.backward()
            # optimizer.step()

            # Compute contrastive loss
            augmented_data = augment_data(tokens)  # Augment data to get a1 and a2
            augmented_reps = encoder(augmented_data)
            augmented_reps, _ = dropout_layer(augmented_reps)
            augmented_logits = classifier(augmented_reps)

            # Compute La: similarity loss between augmented samples and real samples
            similarity_loss = compute_similarity_loss(reps, augmented_reps)

            # Compute Ls: similarity loss between samples with the same label
            same_label_loss = compute_same_label_loss(reps, labels)

            # Compute L1: cross-entropy loss on real samples
            cross_entropy_loss = criterion(augmented_logits, labels)

            # Compute total loss
            total_loss = similarity_loss + same_label_loss + cross_entropy_loss

            total_loss.backward()
            losses.append(total_loss.item())
            optimizer.step()

            # 数据增强
            a1, a2 = augment_data(tokens)

            # 计算特征表示
            reps = encoder(tokens)
            reps_a1 = encoder(a1)
            reps_a2 = encoder(a2)

            logits = classifier(reps)
            loss_ce = criterion(logits, labels)

            # 计算对比度损失 L2
            # 计算 La：增强数据和真实数据的相似性
            loss_La = torch.nn.functional.pairwise_distance(reps, reps_a1) + \
                      torch.nn.functional.pairwise_distance(reps, reps_a2)
            # 计算 Ls：相同标签数据的相似性
            mask = (origin_labels.unsqueeze(1) == origin_labels.unsqueeze(0)).to(torch.float32)
            loss_Ls = torch.nn.functional.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=-1) * mask
            loss_Ls = loss_Ls.sum() / (loss_Ls.sum() != 0).sum()
            # 计算 L1：补充交叉熵损失
            loss_L1 = loss_ce

            # 综合计算总损失
            loss = loss_La + loss_Ls + loss_L1
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        #打印每个训练周期的平均损失
        print(f"loss is {np.array(losses).mean()}")

def augment_data(tokens):
    # Create copies of the input tokens
    a1 = tokens.clone()
    a2 = tokens.clone()

    # Randomly mask some tokens in a1
    mask = torch.rand(a1.size()) < 0.15
    a1[mask] = 0

    # Randomly mask some tokens in a2
    mask = torch.rand(a2.size()) < 0.15
    a2[mask] = 0

    return a1, a2
def compute_similarity_loss(reps, augmented_reps):
    # Normalize the representations and the augmented representations
    reps = F.normalize(reps, p=2, dim=1)
    augmented_reps = F.normalize(augmented_reps, p=2, dim=1)

    # Compute the cosine similarity between the representations and the augmented representations
    similarity = torch.sum(reps * augmented_reps, dim=1)

    # Compute the mean similarity
    mean_similarity = torch.mean(similarity)

    # The loss is the negative mean similarity, because we want to maximize the similarity
    loss = -mean_similarity

    return loss

def compute_same_label_loss(reps, labels):
    # Initialize a list to store the distances
    distances = []

    # Get the unique labels
    unique_labels = torch.unique(labels)

    # For each unique label
    for label in unique_labels:
        # Get the representations of the samples with this label
        same_label_reps = reps[labels == label]

        # If there is only one sample with this label, skip it
        if same_label_reps.shape[0] == 1:
            continue

        # Compute the pairwise distances between the representations
        pairwise_distances = torch.pdist(same_label_reps)

        # Add the mean pairwise distance to the list of distances
        distances.append(pairwise_distances.mean())

    # Compute the mean of the distances
    loss = torch.stack(distances).mean()

    return loss



def get_relation_embeddings(config, encoder, dropout_layer, classifier, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    embeddings = []

    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            embeddings.append(reps.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings