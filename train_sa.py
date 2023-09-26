from torchvision import transforms
from imagenet_dataset import ImageNet_DataSet, AlexNet_Error, ImgNet_C_val_Dst
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet50
import argparse
import torch.nn as nn
import os, torch, random, tqdm, csv
from imagecorruptions import get_corruption_names
import torch.nn.functional as F
import math

print(get_corruption_names("all"))

def parse():
    parser = argparse.ArgumentParser(description="train gurie")
    parser.add_argument("--bs", type=int, default=512, help="batch_size")
    parser.add_argument("--gpuids", type=str, default="0,1,2,3", help="GPU id to train")
    parser.add_argument("--method", type=str, default="R50v_mem", help="name of method")
    parser.add_argument("--mode", type=str, default="train", help="train_or_val")
    parser.add_argument("--root_path", type=str, default="/home/yangzhou/datasets/imagenet/", help="data root path")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/",
                        help="ckpt_path to load/val")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--nb", type=int, default=16, help="number works")
    parser.add_argument("--test_dst", type=str, default="c", help="test datasets")
    arg = parser.parse_args()
    return arg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mCE(error_list):
    mCE = 0.0
    for i, err in enumerate(error_list[:-4]):
        mCE += err / AlexNet_Error[i]
    return mCE / len(AlexNet_Error[:-4]) * 100


@torch.no_grad()
def val_mCE(model, args):
    # model.eval()
    # model.training = False
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
                                weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    epoch, model = load_ckpt(model, optimizer, lr_schedule, args, "last")
    print(epoch)
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = [
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
    test_folder = os.path.join(args.root_path, 'val')
    error_list = []
    fcsv = open("results_c.csv", 'a')
    f_csv = csv.writer(fcsv)
    for crp_idx in tqdm.tqdm(range(-1, len(AlexNet_Error))):
        crp_err = 0.0
        for sev_lvl in range(1, 6):
            acc = 0.0
            if args.test_dst == "s":
                dataset = ImageNet_DataSet(test_folder, test_transform, None, crp_idx, sev_lvl, is_pair=False)
            else:
                ann_path = "/home/yangzhou/datasets/imagenet/meta/val.txt"
                clean_img_path = "/home/yangzhou/datasets/imagenet/ILSVRC2012_img_val/"
                img_path = "/home/yangzhou/datasets/imagenet_c/"
                dataset = ImgNet_C_val_Dst(clean_img_path, img_path, ann_path,
                                           get_corruption_names("all")[crp_idx] if crp_idx != -1 else "clean",
                                           str(sev_lvl))
            test_dataloader = DataLoader(dataset,
                                         batch_size=args.bs,
                                         shuffle=False,
                                         num_workers=args.nb,
                                         pin_memory=True)
            for i, (img, target, _) in enumerate(test_dataloader):
                img, target = img.cuda(), target.cuda()
                _, pred = model(img)
                # pred = model(img)
                # print(F.cross_entropy(pred,target))
                hit = np.count_nonzero(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
                acc += hit
                # print(img.size())
            crp_err += (1 - acc / len(dataset)) * 100
        print(get_corruption_names("all")[crp_idx] if crp_idx != -1 else "clean", ", Error rate is %.2f", crp_err / 5.0)
        error_list.append(crp_err / 5.0)
    error_list = [args.method] + error_list
    f_csv.writerow(error_list)
    fcsv.close()
    del (error_list[0])
    del (error_list[1])
    print("model mCE: %/2f", mCE(error_list))
    return mCE(error_list)


@torch.no_grad()
def val(model, args):
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
    test_folder = os.path.join(args.root_path, 'val')
    dataset = ImageNet_DataSet(test_folder, test_transform, None, 0, 1, is_pair=False)
    test_dataloader = DataLoader(dataset,
                                 batch_size=args.bs,
                                 shuffle=False,
                                 num_workers=args.nb,
                                 pin_memory=True)
    acc = 0.0
    for i, (img, target, _) in tqdm.tqdm(enumerate(test_dataloader)):
        img, target = img.cuda(), target.cuda()
        _, pred, index = model(img)
        # pred = model(img)
        # print(F.cross_entropy(pred,target))
        hit = np.count_nonzero(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
        acc += hit
        # print(img.size())
    return acc / len(dataset)


def load_ckpt(model, optimizer, lr_schedule, args, best_or_last):
    load_path = os.path.join(args.ckpt_path, args.method)
    if not os.path.exists(load_path):
        os.mkdir(load_path)
    if len(os.listdir(load_path)) == 0:
        return -1, model
    load_path = os.path.join(load_path, "%s.pth" % best_or_last)

    state_dict = torch.load(load_path)
    model.module.load_state_dict(state_dict["model"])
    # optimizer.load_state_dict(state_dict["optimizer"])
    lr_schedule.load_state_dict(state_dict["lr_schedule"])
    epoch = state_dict["epoch"]
    # optimizer.state_dict()['param_groups'][0]['lr'] = 1e-4
    print("now lr", optimizer.state_dict()['param_groups'][0]['lr'])
    print("epoch: ", epoch)
    return epoch, model


def save_ckpt(model, epoch, optimizer, lr_schedule, acc, args, is_best=True):
    save_path = os.path.join(args.ckpt_path, args.method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    state_dict = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch
    }
    if is_best:
        state_dict["best_acc"] = acc
        torch.save(state_dict, os.path.join(save_path, "best.pth"))
    else:
        state_dict["acc"] = acc
        torch.save(state_dict, os.path.join(save_path, "last.pth"))


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim=2048, num_embeddings=10000, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        torch.nn.init.uniform_(self.embeddings.weight, 0, 3)

    def forward(self, x, target=None):
        encoding_indices = self.get_code_indices(x, target)
        quantized = self.quantize(encoding_indices)
        # weight, encoding_indices = self.get_code_indices(x)
        # quantized = self.quantize(weight, encoding_indices)

        if not self.training:
            return quantized, encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print("??????????????????",loss)

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, flat_x, target=None):
        # flag = self.training
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = self.embeddings.weight
        weight = F.normalize(weight, p=2, dim=1)
        flag = False
        if flag:
            # print(target.dtype)
            # raise ValueError("target type error! ")
            encoding_indices = target
        else:
            # compute L2 distance
            distances = (
                    torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                    torch.sum(weight ** 2, dim=1) -
                    2. * torch.matmul(flat_x, weight.t())
            )  # [N, M]
            print("hehe")
            # dis, encoding_indices = distances.topk(k=10)
            # index = F.gumbel_softmax(distances, tau=1, hard=False)
            # encoding_indices = torch.argmin(index, dim=1)  # [N,]
            encoding_indices = torch.argmin(distances, dim=1)  # [N,]
            # weight = F.softmax(dis / 2, dim=1)
        return encoding_indices
        # return weight, encoding_indices

    # def quantize(self, weight, encoding_indices):
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        # b, k = weight.size()
        # self.embeddings(encoding_indices)
        # quantized = torch.stack(
        #     [torch.index_select(input=self.embeddings.weight, dim=0, index=encoding_indices[i, :]) for i in range(b)])
        # weight = weight.view(b, 1, k).contiguous()
        # quantized = torch.bmm(weight, quantized).view(b, -1).contiguous()
        # return quantized
        return self.embeddings(encoding_indices)


class new_clsnet(nn.Module):
    def __init__(self, model):
        super(new_clsnet, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = list(model.children())[-1]
        self.codebook = VectorQuantizer(model.fc.in_features, 100000, 0.25)

        self.K = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
        self.Q = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
        self.V = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
        nn.init.kaiming_normal_(self.K)
        nn.init.kaiming_normal_(self.Q)
        nn.init.kaiming_normal_(self.V)

        self.fc_fuse = nn.Sequential(nn.Linear(self.fc.in_features * 2, self.fc.in_features),
                                     nn.ReLU(True))
        # self.mlp = nn.Sequential(nn.Linear(self.fc.in_features, self.fc.in_features),
        #                          nn.BatchNorm1d(self.fc.in_features),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(self.fc.in_features, self.fc.in_features))

    def forward(self, x, target=None):
        bs = x.shape[0]
        x = self.resnet_layer(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if not self.training:
            quantized, index = self.codebook(feat)
            fuse = torch.stack([quantized, feat], dim=2)  # b,d,2
            K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)  # b,d,2
            Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)  # b.d.2
            V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)  # b,d,2
            A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q), dim=1)  # b,2,2
            fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
            fuse = self.fc_fuse(fuse)
            pred = self.fc(fuse)
            # pred = self.fc(quantized)
            # print(quantized, pred)
            # CL_feat = self.mlp(feat)
            return quantized, pred, index

        quantized, e_q_loss, index = self.codebook(feat, target)
        fuse = torch.stack([quantized, feat], dim=2)  # b,d,2
        K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)  # b,d,2
        Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)  # b.d.2
        V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)  # b,d,2
        A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q)/torch.sqrt(torch.tensor(2).to(fuse.device)), dim=1)  # b,2,2
        fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
        fuse = self.fc_fuse(fuse)
        pred = self.fc(fuse)
        # pred = self.fc(quantized)
        ce_loss = F.cross_entropy(pred, target)
        # CL_feat = self.mlp(feat)
        # print("++++++++++++++++++", ce_loss)
        return e_q_loss, ce_loss, index, self.Q


def train_base(model, args):
    max_epoch = 300
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    train_folder = os.path.join(args.root_path, 'train')
    dataset = ImageNet_DataSet(train_folder, train_transform, is_pair=False)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
                                weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    epoch, model = load_ckpt(model, optimizer, lr_schedule, args, best_or_last="last")
    mCE = 0.0
    for ep_ in range(epoch + 1, max_epoch):
        model.train()
        train_dataloader = DataLoader(dataset,
                                      batch_size=args.bs,
                                      shuffle=True,
                                      num_workers=args.nb,
                                      pin_memory=True,
                                      drop_last=False)

        for i, (img, target, _) in enumerate(train_dataloader):
            # print(i, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # img, target = torch.cat([img[:, 0:3, :, :], img[:, 3:, :, :]], dim=0).cuda(), torch.cat([target, target],
            #                                                                                         dim=0).cuda()
            # loss = model(torch.cat([img[:, 0:3, :, :], img[:, 3:6, :, :]], dim=0), target)
            img, target = img.cuda(), target.cuda()
            eq_loss, ce_loss, index, query = model(img, target)
            # print("_______________________________________________",loss)
            loss = (eq_loss + ce_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("epoch: %d, itr: %d, loss: %.6f, eq_loss: %.6f, ce_loss: %.6f" % (
                    ep_, i, loss, eq_loss.mean(), ce_loss.mean()))
                print("index", index[:args.bs // 2])
                print("query", query[0])
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        lr_schedule.step()
        if ep_ % 2 == 0:
            test_mCE = val(model, args)
            print("~~~~~~~~~~~~~~~~~~~~~Acc = %.3f~~~~~~~~~~~~~~~~~~~~~~" % test_mCE)
            if mCE <= test_mCE:
                mCE = test_mCE
                save_ckpt(model, ep_, optimizer, lr_schedule, mCE, args, is_best=True)
            save_ckpt(model, ep_, optimizer, lr_schedule, test_mCE, args, is_best=False)


if __name__ == "__main__":
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids

    model = resnet50(pretrained=True)
    model = new_clsnet(model)
    model = nn.DataParallel(model).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
    for k,v in model.module.named_parameters():
        if k=="K" or k=="Q" or k=="V":
            print(k)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
    #                             weight_decay=5e-4)
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    # epoch, model = load_ckpt(model, optimizer, lr_schedule, args, best_or_last="last")
    # print(model.module.codebook.embeddings.weight)
    # print(model.module.fc.weight)
    if args.mode == "train":
        train_base(model, args)
    else:
        mCE = val(model, args)
