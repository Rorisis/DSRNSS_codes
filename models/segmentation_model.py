import copy
import math
import os
import random
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers.dacs_transforms import get_class_masks, strong_transform, get_dynamic_masks
from helpers.matching_utils import (
    estimate_probability_of_confidence_interval_of_mixture_density, warp)
from helpers.metrics import MyMetricCollection
from helpers.utils import colorize_mask, crop, resolve_ckpt_dir
from PIL import Image
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, instantiate_class

from .heads.base import BaseHead
from .hrda import hrda_backbone, hrda_head
from .modules import DropPath


@MODEL_REGISTRY
class DomainAdaptationSegmentationModel(pl.LightningModule):
    def __init__(self,
                 optimizer_init: dict,
                 lr_scheduler_init: dict,
                 backbone: nn.Module,
                 head: BaseHead,
                 loss: nn.Module,
                 alignment_backbone: Optional[nn.Module] = None,
                 alignment_head: Optional[BaseHead] = None,
                 metrics: dict = {},
                 backbone_lr_factor: float = 1.0,
                 use_refign: bool = False,
                 use_align: bool = True,
                 gamma: float = 0.25,
                 adapt_to_ref: bool = False,
                 disable_M: bool = False,
                 disable_P: bool = False,
                 ema_momentum: float = 0.999,
                 pseudo_label_threshold: float = 0.968,
                 psweight_ignore_top: int = 0,
                 psweight_ignore_bottom: int = 0,
                 enable_fdist: bool = True,
                 fdist_lambda: float = 0.005,
                 fdist_classes: list = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
                 fdist_scale_min_ratio: float = 0.75,
                 color_jitter_s: float = 0.2,
                 color_jitter_p: float = 0.2,
                 blur: bool = True,
                 use_hrda: bool = False,
                 hrda_output_stride: int = 4,
                 hrda_scale_attention: Optional[nn.Module] = None,
                 hr_loss_weight: float = 0.1,
                 use_slide_inference: bool = False,
                 inference_batched_slide: bool = True,
                 inference_crop_size: list = [1080, 1080],
                 inference_stride: list = [420, 420],
                 pretrained: Optional[str] = None,
                 feature_bank: list = [],
                 label_bank: list = [],
                 mask_bank: list = [],
                 step_count: int = 0,
                 step_save: int = 2,
                 ):
        super().__init__()

        #### MODEL ####
        # segmentation
        self.backbone = backbone
        self.head = head
        self.hrda_scale_attention = hrda_scale_attention if use_hrda else None
        # alignment
        self.alignment_backbone = alignment_backbone
        self.alignment_head = alignment_head
        for alignment_m in filter(None, [self.alignment_backbone, self.alignment_head]):
            for param in alignment_m.parameters():
                param.requires_grad = False
        # ema
        self.m_backbone = copy.deepcopy(self.backbone)
        self.m_head = copy.deepcopy(self.head)
        self.m_hrda_scale_attention = copy.deepcopy(self.hrda_scale_attention)
        for param in self.ema_parameters():
            param.requires_grad = False
        # imnet
        self.enable_fdist = enable_fdist
        if self.enable_fdist:
            self.imnet_backbone = copy.deepcopy(self.backbone)
            for param in self.imnet_backbone.parameters():
                param.requires_grad = False

        #### LOSSES ####
        self.loss = loss

        #### METRICS ####
        val_metrics = {'val_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('val', {}).items() for el in m}
        test_metrics = {'test_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('test', {}).items() for el in m}
        self.valid_metrics = MyMetricCollection(val_metrics)
        self.test_metrics = MyMetricCollection(test_metrics)

        #### OPTIMIZATION ####
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.backbone_lr_factor = backbone_lr_factor

        #### REFIGN ####
        self.use_refign = use_refign
        self.use_align = use_align
        self.gamma = gamma
        self.adapt_to_ref = adapt_to_ref
        self.disable_M = disable_M
        self.disable_P = disable_P

        #### OTHER STUFF ####
        self.ema_momentum = ema_momentum
        self.pseudo_label_threshold = pseudo_label_threshold
        self.psweight_ignore_top = psweight_ignore_top
        self.psweight_ignore_bottom = psweight_ignore_bottom
        self.fdist_lambda = fdist_lambda
        self.fdist_classes = fdist_classes
        self.fdist_scale_min_ratio = fdist_scale_min_ratio
        self.color_jitter_s = color_jitter_s
        self.color_jitter_p = color_jitter_p
        self.blur = blur
        self.use_hrda = use_hrda
        if self.use_hrda:
            # apply hrda decorators
            self.backbone.forward = hrda_backbone(
                self.backbone, hrda_output_stride)(self.backbone.forward)
            self.head.forward = hrda_head(
                self.head, self.hrda_scale_attention, hrda_output_stride)(self.head.forward)
            self.m_backbone.forward = hrda_backbone(
                self.m_backbone, hrda_output_stride, is_teacher=True)(self.m_backbone.forward)
            self.m_head.forward = hrda_head(
                self.m_head, self.m_hrda_scale_attention, hrda_output_stride,
                is_teacher=True)(self.m_head.forward)
        self.hr_loss_weight = hr_loss_weight
        self.use_slide_inference = use_slide_inference
        self.inference_batched_slide = inference_batched_slide
        self.inference_crop_size = inference_crop_size
        self.inference_stride = inference_stride
        self.automatic_optimization = False
        self.step_count = step_count
        self.step_save = step_save
        self.feature_bank = feature_bank
        self.label_bank = label_bank
        self.mask_bank = mask_bank

        #### LOAD WEIGHTS ####
        self.load_weights(pretrained)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        self.update_momentum_encoder()

        #
        # SOURCE
        #
        images_src, gt_src = batch['image_src'], batch['semantic_src']
        feats_src = self.backbone(images_src)
        feats_src_ = feats_src
        logits_src = self.head(feats_src)
        if self.use_hrda:
            # feats_src are lr feats
            feats_src = feats_src[0]
            logits_src, hr_logits_src, crop_box_src = logits_src
            logits_src = nn.functional.interpolate(
                logits_src, images_src.shape[-2:], mode='bilinear', align_corners=False)
            cropped_gt_src = crop(gt_src, crop_box_src)
            loss_src = (1 - self.hr_loss_weight) * self.loss(logits_src, gt_src) + \
                self.hr_loss_weight * self.loss(hr_logits_src, cropped_gt_src)
        else:
            logits_src = nn.functional.interpolate(
                logits_src, images_src.shape[-2:], mode='bilinear', align_corners=False)
            loss_src = self.loss(logits_src, gt_src)
        self.log("train_loss_src", loss_src)
        self.manual_backward(loss_src, retain_graph=self.enable_fdist)

        # free graph
        del loss_src
        del logits_src
        if self.use_hrda:
            del hr_logits_src

        # ImageNet feature distance
        if self.enable_fdist:
            loss_featdist_src = self.calc_feat_dist(
                images_src, gt_src, feats_src)
            self.log("train_loss_featdist_src", loss_featdist_src)
            self.manual_backward(loss_featdist_src, retain_graph = True)

            # free graph
            del loss_featdist_src

        #
        # TARGET
        #
        with torch.no_grad():
            if self.adapt_to_ref and random.random() < 0.5:
                adapt_to_ref = True
                images_trg = batch['image_ref']
            else:
                adapt_to_ref = False
                images_trg = batch['image_trg']
            if self.use_refign and not adapt_to_ref:
                images_ref = batch['image_ref']
                b = images_trg.shape[0]
                m_feats_n = self.backbone(images_trg)
                m_input = torch.cat((images_trg, images_ref))
                m_logits = self.m_head(self.m_backbone(m_input))
                m_logits = nn.functional.interpolate(
                    m_logits, size=m_input.shape[-2:], mode='bilinear', align_corners=False)
                m_logits_trg, m_logits_ref = torch.split(
                    m_logits, [b, b], dim=0)
                if self.use_align:
                    warped_m_logits_ref, warp_mask, warp_certs = self.align(
                        m_logits_ref, images_ref, images_trg)
                    m_probs_trg = self.refine(
                        m_logits_trg, warped_m_logits_ref, warp_mask, warp_certs)
                else:
                    m_probs_trg = self.refine(
                        m_logits_trg, m_logits_ref, None, None)
            else:
                m_feats_n = self.backbone(images_trg)
                m_logits_trg = self.m_head(self.m_backbone(images_trg))
                m_logits_trg = nn.functional.interpolate(
                    m_logits_trg, size=images_trg.shape[-2:], mode='bilinear', align_corners=False)
                m_probs_trg = nn.functional.softmax(m_logits_trg, dim=1)
            self.add_features(images_src, feats_src_, gt_src, m_probs_trg)
            mixed_img, mixed_lbl, mixed_weight = self.get_dynamic_mix(
                images_trg, m_probs_trg, images_src, gt_src)

            means, stds = self.get_mean_std()

            if batch_idx % 5 == 0:
                for i in range(len(self.label_bank)):
                    mixed_lbl = self.label_bank[i] * self.mask_bank[i] + (1 - self.mask_bank[i]) * mixed_lbl
                    img_1 = [torch.mul(self.mask_bank[i], self.feature_bank[i][:,j,:,:]) for j in range(self.feature_bank[i].shape[1])]
                    img_1 = torch.stack(img_1, dim = 1)
                    img_2 = [torch.mul((1-self.mask_bank[i]), mixed_img[:,j,:,:]) for j in range(mixed_img.shape[1])]
                    img_2 = torch.stack(img_2, dim = 1)
                    mixed_img = img_1 + img_2

        # Train on mixed images
        mixed_feats = self.backbone(mixed_img)
        mixed_pred = self.head(mixed_feats)
        if self.use_hrda:
            mixed_pred, hr_mixed_pred, crop_box_mixed = mixed_pred
            mixed_pred = nn.functional.interpolate(
                mixed_pred, mixed_img.shape[-2:], mode='bilinear', align_corners=False)
            cropped_mixed_lbl = crop(mixed_lbl, crop_box_mixed)
            cropped_mixed_weight = crop(mixed_weight, crop_box_mixed)
            mixed_loss = (1 - self.hr_loss_weight) * self.loss(mixed_pred, mixed_lbl, pixel_weight=mixed_weight) + \
                self.hr_loss_weight * \
                self.loss(hr_mixed_pred, cropped_mixed_lbl,
                          pixel_weight=cropped_mixed_weight)
            mixed_loss = 0.8 * mixed_loss
        else:
            mixed_pred = nn.functional.interpolate(
                mixed_pred, mixed_img.shape[-2:], mode='bilinear', align_corners=False)
            mixed_loss = self.loss(mixed_pred, mixed_lbl,
                                   pixel_weight=mixed_weight)

        if self.current_epoch >= 8:
            if self.use_hrda:
                feat_s = feats_src[-1]
                label_s_0 = gt_src[0, :, :].unsqueeze(0).contiguous()
                feature_s_0 = feat_s[0, :, :].unsqueeze(0).contiguous()
                label_s_1 = gt_src[1, :, :].unsqueeze(0).contiguous()
                feature_s_1 = feat_s[1, :, :].unsqueeze(0).contiguous()

                label_t = m_probs_trg.clone()
                label_t = nn.functional.softmax(label_t, dim=1)
                label_t = torch.argmax(label_t, dim=1)
                feat_t = m_feats_n[0]
                feat_t = feat_t[-1]
                label_t_0 = label_t[0, :, :].unsqueeze(0).contiguous()
                feature_t_0 = feat_t[0, :, :].unsqueeze(0).contiguous()
                label_t_1 = label_t[1, :, :].unsqueeze(0).contiguous()
                feature_t_1 = feat_t[1, :, :].unsqueeze(0).contiguous()

                feat_m = mixed_feats[0]
                feat_m = feat_m[-1]
                label_m_0 = mixed_lbl[0, :, :].unsqueeze(0).contiguous()
                feature_m_0 = feat_m[0, :, :].unsqueeze(0).contiguous()
                label_m_1 = mixed_lbl[1, :, :].unsqueeze(0).contiguous()
                feature_m_1 = feat_m[1, :, :].unsqueeze(0).contiguous()

                loss_metric_0 = self.metric_loss(feature_s_0, label_s_0, feature_t_0, label_t_0)
                loss_metric_1 = self.metric_loss(feature_s_1, label_s_1, feature_t_1, label_t_1)
                loss_metric = (loss_metric_0 + loss_metric_1) / 2.

                loss_align_0 = self.metric_loss(feature_s_0, label_s_0, feature_m_0, label_m_0)
                loss_align_1 = self.metric_loss(feature_s_1, label_s_1, feature_m_1, label_m_1)
                loss_align = (loss_align_0 + loss_align_1) / 2.
                self.log("loss_metric", loss_metric)
                self.log("loss_align", loss_align)
                loss_metric = 1*(loss_metric + loss_align / 100) 
                self.manual_backward(loss_metric, retain_graph = True)

                del loss_metric
            else:
                feat_s = feats_src[-1]
                label_s_0 = gt_src[0, :, :].unsqueeze(0).contiguous()
                feature_s_0 = feat_s[0, :, :].unsqueeze(0).contiguous()
                label_s_1 = gt_src[1, :, :].unsqueeze(0).contiguous()
                feature_s_1 = feat_s[1, :, :].unsqueeze(0).contiguous()

                label_t = m_probs_trg.clone()
                label_t = nn.functional.softmax(label_t, dim=1)
                label_t = torch.argmax(label_t, dim=1)
                # feat_t = m_feats_n[0]
                feat_t = m_feats_n[-1]
                label_t_0 = label_t[0, :, :].unsqueeze(0).contiguous()
                feature_t_0 = feat_t[0, :, :].unsqueeze(0).contiguous()
                label_t_1 = label_t[1, :, :].unsqueeze(0).contiguous()
                feature_t_1 = feat_t[1, :, :].unsqueeze(0).contiguous()

                # feat_m = mixed_feats[0]
                feat_m = mixed_feats[-1]
                label_m_0 = mixed_lbl[0, :, :].unsqueeze(0).contiguous()
                feature_m_0 = feat_m[0, :, :].unsqueeze(0).contiguous()
                label_m_1 = mixed_lbl[1, :, :].unsqueeze(0).contiguous()
                feature_m_1 = feat_m[1, :, :].unsqueeze(0).contiguous()

                loss_metric_0 = self.metric_loss(feature_s_0, label_s_0, feature_t_0, label_t_0)
                loss_metric_1 = self.metric_loss(feature_s_1, label_s_1, feature_t_1, label_t_1)
                loss_metric = (loss_metric_0 + loss_metric_1) / 2.

                loss_align_0 = self.metric_loss(feature_s_0, label_s_0, feature_m_0, label_m_0)
                loss_align_1 = self.metric_loss(feature_s_1, label_s_1, feature_m_1, label_m_1)
                loss_align = (loss_align_0 + loss_align_1) / 2.
                self.log("loss_metric", loss_metric)
                self.log("loss_align", loss_align)
                loss_metric = loss_metric + loss_align / 10 
                self.manual_backward(loss_metric, retain_graph = True)

                del loss_metric 

        self.log("train_loss_uda_trg", mixed_loss)
        self.manual_backward(mixed_loss)

        # free graph
        del mixed_loss
        del mixed_pred
        if self.use_hrda:
            del hr_mixed_pred

        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['val'][dataloader_idx]
        for k, m in self.valid_metrics.items():
            if src_name in k:
                m(y_hat, y)

    def validation_epoch_end(self, outs):
        out_dict = self.valid_metrics.compute()
        self.valid_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['test'][dataloader_idx]
        for k, m in self.test_metrics.items():
            if src_name in k:
                m(y_hat, y)

    def test_epoch_end(self, outs):
        out_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.trainer.datamodule.predict_on[dataloader_idx]
        save_dir = os.path.join(os.path.dirname(
            resolve_ckpt_dir(self.trainer)), 'preds', dataset_name)
        col_save_dir = os.path.join(os.path.dirname(
            resolve_ckpt_dir(self.trainer)), 'color_preds', dataset_name)
        if self.trainer.is_global_zero:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(col_save_dir, exist_ok=True)
        img_names = batch['filename']
        x = batch['image']
        orig_size = self.trainer.datamodule.predict_ds[dataloader_idx].orig_dims
        y_hat = self.forward(x, orig_size)
        preds = torch.argmax(y_hat, dim=1)
        for pred, im_name in zip(preds, img_names):
            arr = pred.cpu().numpy()
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(save_dir, im_name))
            col_image = colorize_mask(image)
            col_image.save(os.path.join(col_save_dir, im_name))

    def forward(self, x, out_size=None):
        if self.use_slide_inference:
            logits = self.slide_inference(x)
        else:
            logits = self.whole_inference(x)
        if out_size is not None:
            logits = nn.functional.interpolate(
                logits, size=out_size, mode='bilinear', align_corners=False)
        return logits

    def whole_inference(self, x):
        logits = self.head(self.backbone(x))
        logits = nn.functional.interpolate(
            logits, x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def slide_inference(self, img):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.inference_stride
        h_crop, w_crop = self.inference_crop_size
        batched_slide = self.inference_batched_slide
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.head.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.whole_inference(crop_imgs)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.whole_inference(crop_img)
                    preds += nn.functional.pad(crop_seg_logit,
                                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                                int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

    def configure_optimizers(self):
        optimizer = instantiate_class(
            self.optimizer_parameters(), self.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        return [optimizer], [lr_scheduler]

    def optimizer_parameters(self):
        backbone_weight_params = []
        backbone_bias_params = []
        head_weight_params = []
        head_bias_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('backbone'):
                if len(p.shape) == 1:  # bias and BN params
                    backbone_bias_params.append(p)
                else:
                    backbone_weight_params.append(p)
            else:
                if len(p.shape) == 1:  # bias and BN params
                    head_bias_params.append(p)
                else:
                    head_weight_params.append(p)
        lr = self.optimizer_init['init_args']['lr']
        weight_decay = self.optimizer_init['init_args']['weight_decay']
        return [
            {'name': 'head_weight', 'params': head_weight_params,
                'lr': lr, 'weight_decay': weight_decay},
            {'name': 'head_bias', 'params': head_bias_params,
                'lr': lr, 'weight_decay': 0},
            {'name': 'backbone_weight', 'params': backbone_weight_params,
                'lr': self.backbone_lr_factor * lr, 'weight_decay': weight_decay},
            {'name': 'backbone_bias', 'params': backbone_bias_params,
                'lr': self.backbone_lr_factor * lr, 'weight_decay': 0}
        ]

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=self.device)
        elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrain_path)):
            checkpoint = torch.load(os.path.join(os.environ.get(
                'TORCH_HOME', ''), 'hub', pretrain_path), map_location=self.device)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrain_path, progress=True, map_location=self.device)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def refine(self, logits_trg, logits_ref, warp_mask, certs):
        c = logits_trg.shape[1]
        assert c == 19, 'we assume cityscapes classes'

        probs_trg = nn.functional.softmax(logits_trg, dim=1)
        probs_ref = nn.functional.softmax(logits_ref, dim=1)
        pred_trg = torch.argmax(probs_trg, dim=1)  # b x h x w
        pred_ref = torch.argmax(probs_ref, dim=1)  # b x h x w

        # trust score s
        s = torch.mean(self.eta(logits_trg), dim=(1, 2)) ** self.gamma  # b

        # large static class mask M
        static_large_classes = [0, 1, 2, 3, 4, 8, 9, 10]
        static_large_mask_trg = torch.zeros_like(pred_trg, dtype=torch.bool)
        static_large_mask_ref = torch.zeros_like(pred_ref, dtype=torch.bool)
        for c in static_large_classes:
            static_large_mask_trg[pred_trg == c] = True
            static_large_mask_ref[pred_ref == c] = True
        M = static_large_mask_trg & static_large_mask_ref
        M = M.unsqueeze(1).expand_as(probs_trg).clone()
        M[:, 5:8, :, :] = 0  # small static class channels
        M[:, 11:, :, :] = 0  # dynamic class channels

        if self.disable_M:
            M.fill_(0)

        if self.disable_P:
            certs = None

        # P_R
        if certs is not None:
            P = certs.expand_as(probs_trg)
        else:
            P = torch.full_like(probs_trg, 0.5)

        epsilon = s.view(-1, 1, 1, 1) * torch.maximum(P, M)

        if warp_mask is not None:
            warp_mask = warp_mask.unsqueeze(1).expand_as(probs_trg)
            epsilon[~warp_mask] = 0.0  # in area where there is no match

        probs_trg_refined = (1 - epsilon) * probs_trg + epsilon * probs_ref
        return probs_trg_refined

    @staticmethod
    @torch.no_grad()
    def eta(logits):  # normalized entropy / efficiency
        dim = logits.shape[1]
        p_log_p = nn.functional.softmax(
            logits, dim=1) * nn.functional.log_softmax(logits, dim=1)
        ent = -1.0 * p_log_p.sum(dim=1)  # b x h x w
        return ent / math.log(dim)

    @torch.no_grad()
    def align(self, logits_ref, images_ref, images_trg):
        assert self.alignment_head is not None

        b, _, h, w = images_trg.shape
        images_trg_256 = nn.functional.interpolate(
            images_trg, size=(256, 256), mode='area')
        images_ref_256 = nn.functional.interpolate(
            images_ref, size=(256, 256), mode='area')

        x_backbone = self.alignment_backbone(
            torch.cat([images_ref, images_trg]), extract_only_indices=[-3, -2])
        unpacked_x = [torch.split(l, [b, b]) for l in x_backbone]
        pyr_ref, pyr_trg = zip(*unpacked_x)
        x_backbone_256 = self.alignment_backbone(
            torch.cat([images_ref_256, images_trg_256]), extract_only_indices=[-2, -1])
        unpacked_x_256 = [torch.split(l, [b, b]) for l in x_backbone_256]
        pyr_ref_256, pyr_trg_256 = zip(*unpacked_x_256)

        trg_ref_flow, trg_ref_uncert = self.alignment_head(
            pyr_trg, pyr_ref, pyr_trg_256, pyr_ref_256, (h, w))[-1]
        trg_ref_flow = nn.functional.interpolate(
            trg_ref_flow, size=(h, w), mode='bilinear', align_corners=False)
        trg_ref_uncert = nn.functional.interpolate(
            trg_ref_uncert, size=(h, w), mode='bilinear', align_corners=False)

        trg_ref_cert = estimate_probability_of_confidence_interval_of_mixture_density(
            trg_ref_uncert, R=1.0)
        warped_ref_logits, trg_ref_mask = warp(
            logits_ref, trg_ref_flow, return_mask=True)
        return warped_ref_logits, trg_ref_mask, trg_ref_cert

    @torch.no_grad()
    def get_dynamic_mix(self, images_trg, probs_trg, images_src, gt_src):

        # take first source images in batch
        src_images_bs = images_src.shape[0]
        trg_images_bs = images_trg.shape[0]
        if src_images_bs > trg_images_bs:
            images_src = images_src[:trg_images_bs]
            gt_src = gt_src[:trg_images_bs]

        images_bs = images_trg.shape[0]
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
        }
        
        trg_pseudo_prob, trg_pseudo_label = torch.max(probs_trg, dim=1)
        trg_ps_large_p = trg_pseudo_prob.ge(
            self.pseudo_label_threshold).long() == 1
        trg_ps_size = torch.numel(trg_pseudo_label)
        trg_pseudo_weight = torch.sum(trg_ps_large_p) / trg_ps_size
        trg_pseudo_weight = torch.full_like(trg_pseudo_prob, trg_pseudo_weight)
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            trg_pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            trg_pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(
            (trg_pseudo_weight.shape), device=self.device)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * images_bs, [None] * images_bs
        mix_masks = get_dynamic_masks(gt_src, probs_trg)

        for i in range(images_bs):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((images_src[i], images_trg[i])),
                target=torch.stack((gt_src[i], trg_pseudo_label[i])))
            _, trg_pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], trg_pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl).squeeze(1)
        return mixed_img, mixed_lbl, trg_pseudo_weight

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            if self.use_hrda:
                img = nn.functional.interpolate(
                    img, scale_factor=0.5, mode='bilinear', align_corners=False)
            feat_imnet = self.imnet_backbone(img)
            if isinstance(feat_imnet, Sequence):
                feat_imnet = [f.detach() for f in feat_imnet]
            else:
                feat_imnet = [feat_imnet.detach()]
                if feat is not None:
                    feat = [feat]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = self.downscale_label_ratio(gt.unsqueeze(1), scale_factor,
                                                     self.fdist_scale_min_ratio,
                                                     self.head.num_classes,
                                                     255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_loss = self.fdist_lambda * feat_dist
        return feat_loss

    @torch.no_grad()
    def add_features(self, imgs, feats_src, gt_src, probs_trg):
        class_bank = [12,14,15,17,18]
        # feats = feats_src.clone()
        labels = gt_src.clone()
        labels_ = gt_src.clone()
        tailed_labels = torch.zeros_like(labels)
        for c in class_bank:
            tailed_labels[labels == c] = 1
        if self.label_bank is None:
            self.label_bank = torch.mul(tailed_labels, labels_)
        if self.mask_bank is None:
            self.mask_bank = tailed_labels
        elif self.feature_bank is None:
            img_ = [torch.mul(tailed_labels, imgs[:,i,:,:]) for i in range(imgs.shape[1])]
            strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            }
            img_ = strong_transform(strong_parameters, data = img_)  
            img_ = torch.stack(img_, dim = 1) 
        else:
            self.label_bank.append(torch.mul(tailed_labels, labels_))
            self.mask_bank.append(tailed_labels)
            img_ = [torch.mul(tailed_labels, imgs[:,i,:,:]) for i in range(imgs.shape[1])]
            img_ = torch.stack(img_, dim = 1)
            self.feature_bank.append(img_)

        if self.step_count > self.step_save:
            self.label_bank = self.label_bank[1:]
            self.feature_bank = self.feature_bank[1:]
            self.mask_bank = self.mask_bank[1:]
        else:
            self.step_count += 1

    @staticmethod
    def masked_feat_dist(f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
        return torch.mean(pw_feat_dist)

    @staticmethod
    def downscale_label_ratio(gt,
                              scale_factor,
                              min_ratio,
                              n_classes,
                              ignore_index=255):
        assert scale_factor > 1
        bs, orig_c, orig_h, orig_w = gt.shape
        assert orig_c == 1
        trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
        ignore_substitute = n_classes

        out = gt.clone()  # otw. next line would modify original gt
        out[out == ignore_index] = ignore_substitute
        out = nn.functional.one_hot(
            out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, n_classes +
                                   1, orig_h, orig_w], out.shape
        out = nn.functional.avg_pool2d(out.float(), kernel_size=scale_factor)
        gt_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == ignore_substitute] = ignore_index
        out[gt_ratio < min_ratio] = ignore_index
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out

    def metric_loss(self, feature_s, label_s, feature_t, label_t):
        _, ch, feature_s_h, feature_s_w = feature_s.size()
        label_s_resize = nn.functional.interpolate(label_s.float().unsqueeze(0), size=(feature_s_h, feature_s_w))

        _, _, feature_t_h, feature_t_w = feature_t.size()
        label_t_resize = nn.functional.interpolate(label_t.float().unsqueeze(0), size=(feature_t_h, feature_t_w))

        source_list = torch.unique(label_s_resize.float())
        target_list = torch.unique(label_t_resize.float())

        overlap_classes = []
        for i, index in enumerate(torch.unique(label_s_resize)):
            if index in torch.unique(label_t_resize) and index != 255.:
                overlap_classes.append(index.detach())

        source_prototypes = torch.zeros(size=(19, ch)).cuda()
        for i, index in enumerate(source_list):
            if index!=255.:
                fg_mask_s = ((label_s_resize==index)*1).cuda().detach()
                prototype_s = (fg_mask_s*feature_s).squeeze().resize(ch,feature_s_h*feature_s_w).sum(-1)/fg_mask_s.sum()
                source_prototypes[int(index)] = prototype_s

        cs_map = torch.matmul(nn.functional.normalize(source_prototypes, dim = 1), nn.functional.normalize(feature_t.squeeze().resize(ch,feature_t_h*feature_t_w), dim=0))
        cs_map[cs_map==0] = -1
        cosine_similarity_map = nn.functional.interpolate(cs_map.resize(1, 19,feature_t_h, feature_t_w), size=label_t.size()[-2:])
        cosine_similarity_map *= 10

        cross_entropy_weight = torch.zeros(size=(19,1))
        for i, element in enumerate(overlap_classes):
            cross_entropy_weight[int(element)] = 1

        cross_entropy_weight = cross_entropy_weight.cuda()
        prototype_loss = torch.nn.CrossEntropyLoss(weight= cross_entropy_weight, ignore_index=255)

        prediction_by_cs = nn.functional.softmax(cosine_similarity_map, dim=1)
        target_predicted = prediction_by_cs.argmax(dim=1)
        confidence_of_target_predicted = target_predicted.max(dim=1).values
        confidence_mask  = (confidence_of_target_predicted>0.8)*1
        target_predicted[target_predicted==0] = 20
        masked_target_predicted = target_predicted * confidence_mask
        masked_target_predicted[masked_target_predicted==0] = 255
        masked_target_predicted[masked_target_predicted==20] = 0
        masked_target_predicted_resize = nn.functional.interpolate(masked_target_predicted.float().unsqueeze(0), size=(feature_t_h, feature_t_w), mode='nearest')

        label_t_resize_new = label_t_resize.clone().contiguous()
        label_t_resize_new[label_t_resize_new==255] = masked_target_predicted_resize[label_t_resize_new==255]

        target_list2 = torch.unique(label_t_resize_new)

        overlap_classes2 = []
        for i, index in enumerate(torch.unique(label_s_resize)):
            if index in torch.unique(label_t_resize_new) and index != 255.:
                overlap_classes2.append(index.detach())

        target_prototypes = torch.zeros(size=(19, ch)).cuda()
        for i, index in enumerate(target_list):#target_list
            if index!=255.:
                # fg_mask_t = ((label_t_resize_new==index)*1).cuda()
                fg_mask_t = ((label_t_resize==index)*1).cuda().detach()
                prototype_t = (fg_mask_t*feature_t).squeeze().resize(ch,feature_t_h*feature_t_w).sum(-1)/fg_mask_t.sum()
                target_prototypes[int(index)] = prototype_t

        cs_map2 = torch.matmul(nn.functional.normalize(target_prototypes, dim = 1), nn.functional.normalize(feature_s.squeeze().resize(ch,feature_s_h*feature_s_w), dim=0))
        cs_map2[cs_map2==0] = -1
        cosine_similarity_map2 = nn.functional.interpolate(cs_map2.resize(1, 19, feature_s_h, feature_s_w), size=label_s.size()[-2:])
        cosine_similarity_map2 *= 10

        prototype_loss2 = torch.nn.CrossEntropyLoss(ignore_index=255)

        metric_loss1 = prototype_loss(cosine_similarity_map, label_t)
        metric_loss2 = prototype_loss(cosine_similarity_map2, label_s)

        metric_loss = 0.1 * metric_loss1 + 0.1 * metric_loss2
        return metric_loss

    def ema_parameters(self):
        for m in filter(None, [self.m_backbone, self.m_head, self.m_hrda_scale_attention]):
            for p in m.parameters():
                yield p

    def live_parameters(self):
        for m in filter(None, [self.backbone, self.head, self.hrda_scale_attention]):
            for p in m.parameters():
                yield p

    @torch.no_grad()
    def update_momentum_encoder(self):
        m = min(1.0 - 1 / (float(self.global_step) + 1.0),
                self.ema_momentum)  # limit momentum in the beginning
        for param, param_m in zip(self.live_parameters(), self.ema_parameters()):
            if not param.data.shape:
                param_m.data = param_m.data * m + param.data * (1. - m)
            else:
                param_m.data[:] = param_m[:].data[:] * \
                    m + param[:].data[:] * (1. - m)

    def train(self, mode=True):
        super().train(mode=mode)
        for m in filter(None, [self.alignment_backbone, self.alignment_head]):
            m.eval()
        for m in filter(None, [self.m_backbone, self.m_head, self.m_hrda_scale_attention]):
            if isinstance(m, nn.modules.dropout._DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if self.enable_fdist:  # always in eval mode
            self.imnet_backbone.eval()

    def get_mean_std(self):
        mean = [123.675, 116.28, 103.53]
        mean = [torch.as_tensor(mean, device = 'cuda')]
        mean = torch.stack(mean).view(-1, 3, 1, 1)
        std = [58.395, 57.12, 57.375]
        std = [torch.as_tensor(std, device = 'cuda')]
        std = torch.stack(std).view(-1, 3, 1, 1)
        return mean, std


    def denorm(self, img, mean, std):
        return img.mul(std).add(mean) / 255.0
