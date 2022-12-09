"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.mixture import GaussianMixture
import torch


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Visualizer:

    def __init__(self, keys):
        self.wins = {k: None for k in keys}

    def display(self, image, key):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)

        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))

        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class Cluster:

    def __init__(self, ):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().cuda()

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))

            proposal = (dist > 0.5)
            instance_map[proposal] = id

        return instance_map

    def cluster(self, prediction, n_sigma=1, threshold=0.5,
                im_name=None, gt_instance=None, do_plot=False):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1
        mask = seed_map > 0.5
        if mask.sum() > 128:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().cuda()
            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            if do_plot:
                figure = plt.Figure(figsize=(16, 24))
                ax0 = figure.add_subplot(3, 1, 1)
                ax1 = figure.add_subplot(3, 1, 2)
                ax2 = figure.add_subplot(3, 1, 3)

                for ax in (ax0, ax1, ax2):
                    ax.scatter(
                        spatial_emb_masked[0].cpu().numpy(),
                        spatial_emb_masked[1].cpu().numpy(),
                        color='#dddddd',
                        alpha=0.3,
                        zorder=-1
                    )

            for inst_id in range(1, gt_instance.max().item() + 1):
                gt_mask = gt_instance == inst_id
                gt_mask = gt_mask[mask.squeeze()].view(-1)
                if do_plot:
                    ax1.scatter(
                        spatial_emb_masked[0, gt_mask].cpu().numpy(),
                        spatial_emb_masked[1, gt_mask].cpu().numpy(),
                        # color=np.random.rand(3,),
                        label='object_' + str(count),
                        alpha=0.3,
                    )

            while (unclustered.sum() > 128):

                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).bool()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu()

                        # tmp = instance_mask.squeeze() * 255
                        # for inst in instances:
                        #     tmp1 = inst['mask']
                        #     cnt = ((tmp > 0) & (tmp1 > 0)).sum()

                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})

                        count += 1

                        if do_plot:
                            ax0.text(
                                center[0].item(), center[1].item(), str(count),
                                fontsize=20
                            )

                            ax0.scatter(
                                spatial_emb_masked[0, proposal].cpu().numpy(),
                                spatial_emb_masked[1, proposal].cpu().numpy(),
                                # color=np.random.rand(3,),
                                label='object_' + str(count),
                                alpha=0.3,
                            )

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

            if len(instances) > 0:

                centers, instance_map, instances = self.gmm_refine_clustering(
                    instance_map,
                    instances,
                    spatial_emb,
                    mask,
                    sigma, n_sigma
                )

                if do_plot:
                    n_instances = instance_map.max()
                    for i in range(1, n_instances + 1):
                        ax2.text(
                            centers[i - 1, 0].item(),
                            centers[i - 1, 1].item(), str(i),
                            fontsize=20
                        )
                        spatial_emb_flatten = spatial_emb.permute(1, 2, 0).view(-1, 2)
                        inst_mask = (instance_map == i).view(-1)
                        ax2.scatter(
                            spatial_emb_flatten[inst_mask, 0].cpu().numpy(),
                            spatial_emb_flatten[inst_mask, 1].cpu().numpy(),
                            # color=np.random.rand(3,),
                            label='object_' + str(count),
                            alpha=0.3,
                        )

            if do_plot:
                figure.savefig(f'tmp/{im_name}')

        return instance_map, instances

    def get_instance_map(self, spatial_emb, sigma, n_sigma, mask,
                         seed_emb, seed_scores: List[float]):
        """
        sigma: n_sigma x h x w
        """
        _, height, width = spatial_emb.size()
        instance_map = torch.zeros(height, width).byte()
        instances = []

        spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
        sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)

        instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

        for i, (center, score) in enumerate(zip(seed_emb, seed_scores)):
            # print('i = ', i)
            center = center[:, None]
            seed = torch.argmin(((spatial_emb_masked - center) ** 2).sum(0))
            s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                      center, 2) * s, 0, keepdim=True))

            proposal = (dist > 0.5).squeeze()
            # print('proposal', proposal.sum())
            instance_map_masked[proposal.squeeze()] = i + 1
            # print(instance_map_masked.max(), 'qqq')
            instance_mask = torch.zeros(height, width).bool()
            instance_mask[mask.squeeze().cpu()] = proposal.cpu()

            instances.append(
                {'mask': instance_mask.squeeze() * 255, 'score': score})

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        #     print(instance_map.max(), 'asfd')
        # print(instance_map.max())
        # print(len(instances))
        return instance_map, instances

    def gmm_refine_clustering(self,
                              instance_map,
                              instances,
                              spatial_emb,
                              mask, sigma, n_sigma):
        """
        instance_map: (h, w)
        instances: List[dict]
        spatial_emb: (2, h, w)
        mask: (1, h, w)
        """
        mask_flatten = mask.squeeze().view(-1).cpu().numpy()  # (h, w)
        spatial_emb_flatten = spatial_emb.permute(1, 2, 0).view(-1, 2).cpu().numpy()
        instance_map_flatten = instance_map.view(-1).cpu().numpy()
        X = spatial_emb_flatten[mask_flatten]
        max_id = instance_map_flatten.max()
        centroids = [spatial_emb_flatten[instance_map_flatten == i].mean(0) for i in
                     range(1, max_id + 1)]
        centroids = np.stack(centroids, axis=0)
        gmm = GaussianMixture(n_components=max_id, means_init=centroids, max_iter=10)
        gmm.fit_predict(X)
        centers = torch.tensor(gmm.means_, device=spatial_emb.device)
        instance_map, instances = self.get_instance_map(
            spatial_emb, sigma, n_sigma, mask, centers,
            [d['score'] for d in instances]
        )
        return centers, instance_map, instances


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)
