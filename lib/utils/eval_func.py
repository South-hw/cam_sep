import torch
import torch.nn.functional as F
import torch.linalg as linalg
from torch.utils.data import DataLoader


def extract_features(net, dl):
    features = []
    labels = []

    with torch.no_grad():
        for i, (img, cls) in enumerate(dl):
            img = img.cuda()
            cls = cls.cuda()

            mac_x, gem_x, _ = net.forward_proxy(img)
            embeddings = torch.cat((mac_x, gem_x), dim=1)
            features.append(embeddings)
            labels.append(cls)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        dist = torch.zeros(size=(features.shape[0], features.shape[0])).float().cuda()
        for i in range(features.shape[0]):
            dist[i, :] = F.cosine_similarity(features[i, :].view(1, -1),
                    features, dim=1)

    return dist, labels


def evaluate(dist, labels, ranks=(1, 2, 4, 8)):
    recalls = []
    diag = torch.range(start=0, end=dist.shape[0] - 1).long()
    dist[diag, diag] = -1.0

    for k in ranks:
        ind = dist.topk(k)[1]

        mask = (labels.view(-1, 1) == labels[ind]).sum(dim=1)
        correct = (mask > 0).sum()
        r = correct / dist.shape[0]
        recalls.append(r.cpu().item())

    return recalls


def eval(net, dl):
    dist, labels = extract_features(net, dl)
    recalls = evaluate(dist, labels)

    return recalls

