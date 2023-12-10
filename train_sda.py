from statistics import mean

import sklearn
import torch
import yaml
from tqdm import tqdm
import torch.nn.functional as F
# from imblearn.over_sampling import RandomOverSampler
# import torch.nn.utils as utils
# from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from sda_utils import mmd
# from MMD import MMDLoss
from domain_adaption_net import Net
# from plot_data import get_pca_data, plot_pca_data, plot_pca_data_cdca
# from pre_procesing.train_reduce_dim import pre_processing
from sda_utils import get_cdca_term, mmd, dsne_loss, ccsa_loss, calculate_f1_score, \
    calculate_auc_score, get_one_hot_encoding
import numpy as np
from sda_datasets import TargetDataset, CdcaDataset, ForeverDataIterator  # , SrcDataset, ForeverDataIterator

# from sinkhorn import SinkhornSolver

from torch import nn
import random


# from sinkhorn import SinkhornSolver


def get_absulute_gradient(net):
    gradients = []
    for param in net.parameters():
        gradients.append(param.grad)

    gradient_norm = torch.sqrt(sum([torch.norm(grad) ** 2 for grad in gradients[:-2]]))
    return gradient_norm


def train_sda(src_dataloader, target_dataloader, net, optimizer, criterion, lr_scheduler, train_size,
              loss_type, writer, gpu_flag=False, epoch=20):
    cdca_dataloader_forever = ForeverDataIterator(src_dataloader)
    # target_dataloader_forever = ForeverDataIterator(target_dataloader)
    loss_type = loss_type
    # sinkhorn = SinkhornSolver()
    # mmd = MMDLoss()
    # net.to("cpu")
    for index in tqdm(range(epoch)):
        net.train()
        gradients_epoch = []
        counter = 0
        total_loss = 0
        epoch_loss = []
        while counter < len(src_dataloader):
            data, labels, batch_id = next(cdca_dataloader_forever)
            # target_data, target_labels = next(target_dataloader_forever)
            mask_0 = batch_id == 0
            mask_1 = batch_id == 1

            src_data = data[mask_0].detach()
            target_data = data[mask_1].detach()

            src_labels = labels[mask_0].detach()
            target_labels = labels[mask_1].detach()

            class_number = 7  # max(len(np.unique(src_labels)), len(np.unique(target_labels)))
            src_data, target_data = src_data.clone().detach(), target_data.clone().detach()

            src_labels_encoding = torch.nn.functional.one_hot(src_labels.to(torch.int64), class_number)
            target_labels_encoding = torch.nn.functional.one_hot(target_labels.to(torch.int64), class_number)
            # src_labels = torch.clamp(src_labels, min=1e-3, max=1 - 1e-3).type(torch.float32).detach()
            # target_labels = torch.clamp(target_labels, min=1e-3, max=1 - 1e-3).type(torch.float32).detach()

            src_pred, src_feature = net(src_data.float())
            tgt_pred, tgt_feature = net(target_data.float())
            src_pred = src_pred.type(torch.float32)
            tgt_pred = tgt_pred.type(torch.float32)

            # -----Explicit losses-----
            u = mask_0.sum() / mask_0.shape[0]
            loss_s =  criterion(src_pred, src_labels_encoding.type(torch.float32))
            loss_t = criterion(tgt_pred, target_labels_encoding.type(torch.float32))
            # loss_uda = mmd(src_feature, tgt_feature)
            if loss_type == "source":
                loss = loss_s / loss_s.item()
                loss_value = loss_s.item()
            if loss_type == "target":
                loss = loss_t / loss_t.item()
                loss_value = loss_t.item()

            if loss_type == "s&t":
                loss = loss_t / loss_t.item() + loss_s / loss_s.item()
                loss_value = loss_t.item() + loss_s.item()

            if loss_type == "s&t&u":
                loss_uda = mmd(src_feature, tgt_feature)

                loss = loss_t / loss_t.item() + loss_s / loss_s.item() + loss_uda / loss_uda.item()
                loss_value = loss_t.item() + loss_s.item() + loss_uda.item()

            if loss_type == "s&t&u&c":
                loss_uda = mmd(src_feature, tgt_feature)
                # t_loss = get_cdca_t(src_feature, tgt_feature, src_labels, target_labels,
                #                           n_classes=2)
                labels_t_s, labels_s_t, labels_s_s, labels_t_t = get_cdca_term(src_feature, tgt_feature,
                                                                               src_labels_encoding,
                                                                               target_labels_encoding,
                                                                               n_classes=2)
                loss_t0_cdca = nn.CrossEntropyLoss()(labels_s_t.type(torch.float32).squeeze(), src_pred)
                loss_t1_cdca = nn.CrossEntropyLoss()(labels_t_s.type(torch.float32).squeeze(), tgt_pred)
                loss_cdca = u * loss_t0_cdca + (1 - u) * loss_t1_cdca
                loss = (1-u) *loss_t / loss_t.item() + u*loss_s/loss_s.item() + loss_cdca  # + loss_uda  + 10*loss_cdca  # loss_t / loss_t.item() + loss_s / loss_s.item() + 0 * loss_uda / loss_uda.item() + 0 * loss_cdca / loss_cdca.item()
                loss_value = loss_t.item() + loss_s.item() + +  loss_uda.item() + loss_cdca.item()  # + 10 * loss_cdca.item()  #
                # +loss_uda.item()# + loss_uda.item()

            if loss_type == 'dSNE':
                loss_dsne = dsne_loss(src_feature, src_labels, tgt_feature, target_labels)
                loss = loss_s / loss_s.item() + loss_t / loss_t.item() + loss_dsne / loss_dsne.item()
                loss_value = loss_s.item() + loss_t.item() + loss_dsne.item()

            if loss_type == 'CCSA':
                loss_csca = ccsa_loss(src_feature, tgt_feature,
                                      (src_labels == target_labels).float())
                loss = loss_s / loss_s.item() + loss_csca / loss_csca.item()
                loss_value = loss_s.item() + loss_csca.item()

            counter += 1
            epoch_loss.append(loss_value)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            gradients_epoch.append(get_absulute_gradient(net))

        total_loss = mean(epoch_loss)
        # print(f"gradient: {max(gradients_epoch)}")
        # print(total_loss)
        writer.add_scalar(f"Loss/{loss_type}/train-{train_size}", total_loss, global_step=index)
        writer.add_scalar(f"Graidents/{loss_type}/train-{train_size}", max(gradients_epoch), global_step=index)
        lr_scheduler.step()
    # net.to('cpu')
    return net


def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class, _ in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        if count_per_class[i] != 0:
            weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class, batch_id) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights


def cdca_alignment(config, adata1, adata2, embed='', resample_data="resample"):
    writer = SummaryWriter(log_dir=f'runs/{config["experiment_name"]}')

    test_size = config["test_size"]
    if embed != '':
        src_data_without_labels, target_data_without_labels = adata1.obsm[embed], adata2.obsm[embed]
    else:
        src_data_without_labels, target_data_without_labels = adata1.X, adata2.X

    labels_src = np.array(adata1.obs["celltype"])
    labels_target = np.array(adata2.obs["celltype"])

    batch_id = np.concatenate((np.array([0] * len(labels_src)), np.array([1] * len(labels_target))))
    labels = np.concatenate((labels_src, labels_target))
    cells = np.concatenate((src_data_without_labels, target_data_without_labels), axis=0)

    print(f"number of celltype src:{np.unique(labels_src)}")
    print(f"number of celltype target:{np.unique(labels_target)}")

    class_number_src = len(np.unique(labels_src))
    class_number_target = len(np.unique(labels_target))
    class_number = max(class_number_src, class_number_target)
    cdca_dataset = CdcaDataset(cells, labels, batch_id, class_number)  # , class_number )# , method=resample_data)
    # target_dataset = TargetDataset(target_data_without_labels, labels_target)

    generator = torch.Generator()
    generator.manual_seed(0)
    cdca_test, cdca_train = torch.utils.data.random_split(cdca_dataset, [int(len(
        cdca_dataset) * test_size),
                                                                         len(cdca_dataset) - int(
                                                                             len(cdca_dataset) * test_size)]
                                                          , generator=generator)
    weights_cdca = make_weights_for_balanced_classes(cdca_train, class_number)
    weights_cdca = torch.DoubleTensor(weights_cdca)
    sampler_cdca = torch.utils.data.sampler.WeightedRandomSampler(weights_cdca, len(weights_cdca))

    # train_set_target_dataset = TargetDataset(train_data, train_labels)  # , class_number)
    cdca_dataloader = DataLoader(dataset=cdca_train, batch_size=int(config["batch_size"]), sampler=sampler_cdca,
                                 drop_last=True)

    criterion = nn.CrossEntropyLoss()

    net = Net(config["input_dim"],
              config["hidden_dim"],
              config["drop_prob"],
              config["hidden_layers"], class_num=class_number)

    optimizer = torch.optim.Adam(net.parameters()
                                 , lr=config["lr"], weight_decay=0.2)
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    net = train_sda(cdca_dataloader, None, net, optimizer, criterion,
                    lr_scheduler, train_size=1 - config["test_size"],
                    loss_type=config["loss_type"],
                    epoch=config["epochs"], writer=writer)

    f1_score = calculate_f1_score(cdca_test, net)
    # auc_score = calculate_auc_score(test_set_target_dataset, net)
    print(f"f1_score: {f1_score}")
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"auc_score: {auc_score}")

    writer.flush()

    writer.close()
