import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import random
from model.ACA_attention import ACAAttention
import numpy as np


class LayerNorm2d(nn.Module):  # Used for normalizing features
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)          # Calculate the mean
        s = (x - u).pow(2).mean(1, keepdim=True)  # Calculate variance
        x = (x - u) / torch.sqrt(s + self.eps)    # Normalize x
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # Multiply by weight and add bias
        return x


class QSM(nn.Module):  # Query Self-similar Matching for CD-FSS
    def __init__(self, backbone, dim_ls, local_noise_std=0.75):
        super(QSM, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        # Extract features through ResNet50 and perturb the first 4 convolutional blocks (a total of 5) with features
        self.layer_block = [self.layer0, self.layer1, self.layer2, self.layer3]

        self.local_noise_std = local_noise_std

        self.perturb_layers_idx = {}     # Storage perturbation layer index
        self.perturb_layers = [0, 1, 2]  # Layers that require disturbance
        self.perturb_layer_dim = dim_ls  # Input/Output channels for each perturbation layer, dim_ls = [128, 256, 512]

        self.DR_Adapter = nn.ModuleList()  # DRAM (Domain Rectifying Adapter Module)

        for idx, layer in enumerate(self.perturb_layers):  # Traverse each perturbation layer
            self.perturb_layers_idx[layer] = idx
            Adapter = nn.Sequential(
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),                     # Using 7x7 dilated convolution to correct features
                LayerNorm2d(self.perturb_layer_dim[idx]),  # Normalize the features
                nn.ReLU(),
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),
                LayerNorm2d(self.perturb_layer_dim[idx]),
                nn.ReLU(),
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),
            )
            self.DR_Adapter.append(Adapter)

        # global_importance: g1=2-g2, g3=2-g4
        self.g2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.g2.data.fill_(1.0)
        self.g4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.g4.data.fill_(1.0)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q, training=True, global_noise_ls=[], datum_center_ls=[], get_prot=False):

        # ################################################# DRAM begins ################################################
        # len(img_s_list)  = 1
        # len(mask_s_list) = 1
        # img _s_list[0].shape: ([bsz, 3, 400, 400])
        # mask_s_list[k].shape:      ([1, 400, 400])

        h, w = img_q.shape[-2:]  # Get the height and width of the query image
        if training:  # perturbations are randomly initiated with probability p during training
            p_local = random.random()
            p_global = random.random()
        else:
            p_local = 1  # When testing, p=1, consider all features as perturbed features
            p_global = 1

        feature_s_list = []  # feature maps of support images

        x_params = []  # Store some intermediate calculation results for calculating loop alignment loss
        s_0 = img_s_list[0]  # ([bsz, 3, 400, 400]) Use the first feature map as the supporting feature map
        q_0 = img_q          # ([bsz, 3, 400, 400])

        x_layer_style = []
        x_ori_vec = []
        for idx, layer in enumerate(self.layer_block):  # idx is a multi-layer feature map index
            s_0 = layer(s_0)  # Extract features through ResNet50, inject noise into the first four convolutional blocks
            q_0 = layer(q_0)  # Extract features through ResNet50, inject noise into the first four convolutional blocks

            # Initial feature map shape of each layer：
            # q_0.shape: ([bsz, 128, 100, 100])
            # q_0.shape: ([bsz, 256, 100, 100])
            # q_0.shape: ([bsz, 512,  50,  50])
            # q_0.shape: ([bsz, 1024, 50,  50])

            # domain rectifying
            # detach: Convert the result into a constant to avoid it being updated by gradient backpropagation
            if idx in self.perturb_layers:  #
                x_q_mean = q_0.mean(dim=(2, 3), keepdim=True).detach()  # Calculate the initial mean
                x_s_mean = s_0.mean(dim=(2, 3), keepdim=True).detach()
                x_q_var = q_0.var(dim=(2, 3), keepdim=True).detach()    # Calculate initial variance
                x_s_var = s_0.var(dim=(2, 3), keepdim=True).detach()
                x_ori_mean = torch.cat((x_q_mean, x_s_mean), dim=0)  # to form a style mean
                x_ori_var = torch.cat((x_q_var, x_s_var), dim=0)     # to form a style variance
                x_layer_style.append([x_ori_mean, x_ori_var])  # Save the current style for updating the global style

                if get_prot:  # If a prototype is passed in, skip it
                    continue

                x_q_disturb = q_0  # Initialize the perturbation query feature to q_0
                x_s_disturb = s_0  # Initialize the disturbance support feature to s_0
                flag = 0  # Domain perturbation flag, flag==1 indicates that feature perturbation has been performed
                if training:  # Perform perturbation with probability p
                    if p_local < 0.5:  # local disturbance
                        x_q_disturb, x_s_disturb, alpha, beta = self.local_perturb(x_q_disturb, x_s_disturb)
                        flag = 1
                    if p_global < 0.5:  # Global perturbation while returning global style momentum
                        global_noise_alpha, global_noise_beta = global_noise_ls[self.perturb_layers_idx[idx]]
                        x_q_disturb, x_s_disturb, datum_center, datum_var = self.global_perturb(x_q_disturb,
                                                                                                x_s_disturb,
                                                                                                global_noise_alpha,
                                                                                                global_noise_beta,
                                                                                                datum_center_ls, idx,
                                                                                                x_layer_style)
                        flag = 1

                # Calculate the mean miu and variance sigma of the perturbed feature map, and store them
                x_q_disturb_miu = x_q_disturb.mean(dim=(2, 3), keepdim=True)
                x_q_disturb_sigma = x_q_disturb.var(dim=(2, 3), keepdim=True)
                x_s_disturb_miu = x_s_disturb.mean(dim=(2, 3), keepdim=True)
                x_s_disturb_sigma = x_s_disturb.var(dim=(2, 3), keepdim=True)
                dist_statistics = (x_q_disturb_miu, x_q_disturb_sigma, x_s_disturb_miu, x_s_disturb_sigma)

                # During the verification or testing phase, perform 'domain calibration'
                if (flag == 1) or (training == False):
                    x_q_rectify, x_s_rectify, x_q_rect_miu, x_q_rect_sigma, x_s_rect_miu, x_s_rect_sigma = self.domain_rectify(x_q_disturb, x_s_disturb, idx, dist_statistics)
                    q_0 = x_q_rectify  # Save the corrected features as q_0 and s_0
                    s_0 = x_s_rectify

                # additional "cyclic_rectify" is performed to improve alignment with the original features
                if training:
                    # Perform feature correction, calculate the mean and variance for calculating the joint loss
                    if flag == 1:  # If there is a disturbance (which must be corrected), perform cyclic correction
                        if p_local < 0.5:  # Perform local correction and calculate the mean and variance
                            second_rect_q_miu, second_rect_q_sigma = self.cyclic_rectify(idx, alpha, beta, x_q_rect_miu, x_q_rect_sigma, q_0)
                            second_rect_s_miu, second_rect_s_sigma = self.cyclic_rectify(idx, alpha, beta, x_s_rect_miu, x_s_rect_sigma, s_0)
                        elif p_global < 0.5:  # Perform global correction and calculate the mean and variance
                            second_rect_q_miu, second_rect_q_sigma = self.cyclic_rectify(idx, global_noise_alpha, global_noise_beta, x_q_rect_miu, x_q_rect_sigma, q_0)
                            second_rect_s_miu, second_rect_s_sigma = self.cyclic_rectify(idx, global_noise_alpha, global_noise_beta, x_s_rect_miu, x_s_rect_sigma, s_0)

                            x_ori_mean = torch.cat((datum_center, datum_center), dim=0)  # cat style mean
                            x_ori_var = torch.cat((datum_var, datum_var), dim=0)         # cat style variance

                        second_rect_mean = torch.cat((second_rect_q_miu, second_rect_s_miu), dim=0)      # cat mean
                        second_rect_sigma = torch.cat((second_rect_q_sigma, second_rect_s_sigma),dim=0)  # cat variance

                        first_rect_mean = torch.cat((x_q_rect_miu, x_s_rect_miu), dim=0)                 # cat mean
                        first_rect_sigma = torch.cat((x_q_rect_sigma, x_s_rect_sigma), dim=0)            # cat variance

                        x_param = (x_ori_mean, x_ori_var, first_rect_mean, first_rect_sigma, second_rect_mean, second_rect_sigma)
                        x_params.append(x_param)

                        # The x_param stores:
                        # (x_ori_mean, x_ori_var, first_rect_mean, first_rect_var, second_rect_mean, second_rect_var)
                        # (query and support) original style mean,
                        # (query and support) original style variance,
                        # (query and support) domain corrected feature mean,
                        # (query and support) domain corrected feature variance,
                        # (query and support) cyclically corrected feature mean,
                        # (query and support) cyclically corrected feature variance.

        # ################################################# DRAM ends ##################################################

        # ###################################### Significant Area Detection begins #####################################

        feature_q = q_0             # Save (domain corrected) query feature
        feature_s_list.append(s_0)  # Save (domain corrected) support features
        # q_0.shape [bsz, 1024, 50, 50]
        # s_0.shape [bsz, 1024, 50, 50]

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []  # Storage supports image foreground features (for computing prototypes)
        feature_bg_list = []  # Storage supports image background features (for computing prototypes)
        supp_out_ls = []      # Storage similarity output

        # Calculate the self similarity mask for the currently supported feature maps and save it as suppl_out
        for k in range(len(img_s_list)):
            # feature_s_list[k].shape： [1, 1024, 50, 50]
            # mask_s_list[k].shape：    [1, 400, 400]
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)  # [None, :] Add a bsz dimension to ensure the correct shape
            feature_bg_list.append(feature_bg)
            if self.training:  # If training, its necessary to calculate the overall style similarity
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0
                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        Ps_f = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        Ps_b = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        # Ps_f.shape [1, 1024, 1, 1]
        # Ps_b.shape [1, 1024, 1, 1]

        # Matching the similarity to obtain significantly similar area
        significant_area = self.similarity_func(feature_q, Ps_f, Ps_b)

        # ###################################### Significant Area Detection ends #######################################

        # ################################################# QSM begins #################################################

        # SSP: Self-Support Matching
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, significant_area)
        Pf_1 = SSFP_1  # [bsz, 1024, 1, 1]
        # __ = ASFP_1  # [bsz, 1024, 1, 1]  ### We do not use it !!!
        Pb_1 = SSBP_1  # [bsz, 1024, 1, 1]
        Pb_2 = ASBP_1  # [bsz, 1024, 50, 50]

        # FSM: Foreground Self-similar Matching
        M_self = self.self_similarity(feature_q, Pf_1)
        target_size = (400, 400)  # interpolate M_self to size [bsz, 400, 400]
        M_self = F.interpolate(M_self.unsqueeze(1), size=target_size, mode='bilinear', align_corners=True).squeeze(1)
        Pf_2 = self.masked_average_pooling(feature_q, (M_self == 1).float())[None, :]
        Pf_2 = torch.mean(Pf_2, dim=0).unsqueeze(-1).unsqueeze(-1)

        # ACAM: Adjacent Channel Attention Module (local)
        ACAM = ACAAttention(channel=1024).cuda()
        Fq_w = ACAM(feature_q)

        # APFM: Adaptive Prototype Fusion Module. Fusing prototypes based on their contributions & global importance
        contribution1, contribution2, contribution3, contribution4 = self.contribution_calculate(Fq_w, Pf_1, Pf_2, Pb_1, Pb_2)

        P1 = Pf_1 * contribution1.view(-1, 1, 1, 1) * (2 - self.g2)  # [bsz, 1024, 1, 1]    g1 = 2 - g2
        P2 = Pf_2 * contribution2.view(-1, 1, 1, 1) * self.g2        # [bsz, 1024, 1, 1]
        P3 = Pb_1 * contribution3.view(-1, 1, 1, 1) * (2 - self.g4)  # [bsz, 1024, 1, 1]    g3 = 2 - g4
        P4 = Pb_2 * contribution4.view(-1, 1, 1, 1) * self.g4        # [bsz, 1024, 50, 50]
        Pf = P1 + P2
        Pb = P3 + P4

        # Perform final matching to get the query mask
        out_1 = self.similarity_func(Fq_w, Pf, Pb)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)  # dual channel similarity matrix
        out_ls = [out_1]

        if self.training:  # Calculate the similarity between query features and query prototypes
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)
            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0
            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)
            out_ls.append(self_out)  # Save the 'self similarity matrix of query image', to calculate the loss
            out_ls.append(supp_out)  # Save the 'self similarity matrix of support images', to calculate the loss

        out_ls.append(x_ori_vec)      # Original Style Momentum
        out_ls.append(x_params)       # Parameters used for cyclic alignment loss
        out_ls.append(x_layer_style)  # Original mean and original variance before disturbance

        # out_1: Dual channel similarity matrix, (background similarity, foreground similarity), ([bsz, 2, 400, 400]),
        # self_out: The similarity between query feature and query prototype    : ([bsz, 2, 400, 400])
        # supp_out: The similarity between support feature and support prototype: ([bsz, 2, 400, 400])
        # x_ori_vec: Original Style Momentum, [null]
        # x_params: (x_ori_mean, x_ori_var, first_rect_mean, first_rect_sigma, second_rect_mean, second_rect_sigma)
        # x_layer_style: Original mean and original variance before disturbance

        # while trn,  out_ls: [out_1, self_out, supp_out, x_ori_vec, x_params, x_layer_style]
        # while test, out_ls: [out_1, x_ori_vec, x_params, x_layer_style]

        # ################################################# QSM ends ###################################################

        return out_ls, M_self

    def local_perturb(self, x_q_disturb, x_s_disturb):  # Implement feature perturbation through alpha and beta
        zeros_mat = torch.zeros(x_q_disturb.mean(dim=(2, 3), keepdim=True).shape)  # Create a 0 matrix
        ones_mat = torch.ones(x_q_disturb.mean(dim=(2, 3), keepdim=True).shape)    # Create a 1 matrix
        alpha = torch.normal(zeros_mat, self.local_noise_std * ones_mat).cuda()  # Sample the perturbation factor alpha
        beta = torch.normal(zeros_mat, self.local_noise_std * ones_mat).cuda()   # Sample the perturbation factor beta

        local_x_q_disturb = ((1 + alpha) * x_q_disturb - alpha * x_q_disturb.mean(dim=(2, 3), keepdim=True)
                             + beta * x_q_disturb.mean(dim=(2, 3), keepdim=True))  # Disturbance query features
        local_x_s_disturb = ((1 + alpha) * x_s_disturb - alpha * x_s_disturb.mean(dim=(2, 3), keepdim=True)
                             + beta * x_s_disturb.mean(dim=(2, 3), keepdim=True))  # Disturbance support features

        return local_x_q_disturb, local_x_s_disturb, alpha, beta

    def global_perturb(self, x_q_disturb, x_s_disturb, global_noise_alpha, global_noise_beta, datum_center_ls, idx,
                       x_layer_style):
        if len(datum_center_ls) > 0:  # If datum_center_ls is not empty, get the style mean and style variance
            datum_center, datum_var = datum_center_ls[self.perturb_layers_idx[idx]]
        else:  # If datum_center_ls is empty, read the data center and variance of x_layer_style, as the style
            datum_center, datum_var = x_layer_style[self.perturb_layers_idx[idx]]

            datum_center = datum_center.mean(dim=0, keepdim=True)  # Update style mean by calculating the average
            datum_var = datum_var.mean(dim=0, keepdim=True)        # Update style variance by calculating the average

        # Extend the bsz of style mean and style variance to be the same as x_q_disturb and x_s_disturb
        datum_center = datum_center.repeat(len(x_q_disturb), 1, 1, 1).detach()
        datum_var = datum_var.repeat(len(x_s_disturb), 1, 1, 1).detach()

        # Perform global feature perturbation
        global_x_s_disturb = (1 + global_noise_alpha) * x_s_disturb - global_noise_alpha * datum_center + global_noise_beta * datum_center
        global_x_q_disturb = (1 + global_noise_alpha) * x_q_disturb - global_noise_alpha * datum_center + global_noise_beta * datum_center

        return global_x_q_disturb, global_x_s_disturb, datum_center, datum_var  # Return feature&style after disturbance

    def domain_rectify(self, x_q_disturb, x_s_disturb, idx, dist_statistics):
        x_q_disturb_miu, x_q_disturb_sigma, x_s_disturb_miu, x_s_disturb_sigma = dist_statistics
        # Obtain the mean and variance of perturbed (query and support) features

        # Correct query features through DR-Adapter and calculate the corrected mean and variance
        x_q_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](x_q_disturb.detach())
        x_q_rect_beta = x_q_rect.mean(dim=(2, 3), keepdim=True)
        x_q_rect_alpha = x_q_rect.var(dim=(2, 3), keepdim=True)
        x_q_rect_miu = (1 + x_q_rect_beta) * x_q_disturb_miu
        x_q_rect_sigma = (1 + x_q_rect_alpha) * x_q_disturb_sigma
        x_q_rectify = ((x_q_disturb - x_q_disturb_miu) / (1e-6 + x_q_disturb_sigma)) * x_q_rect_sigma + x_q_rect_miu

        # Correct support features through DR-Adapter and calculate the corrected mean and variance
        x_s_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](x_s_disturb.detach())
        x_s_rect_beta = x_s_rect.mean(dim=(2, 3), keepdim=True)
        x_s_rect_alpha = x_s_rect.var(dim=(2, 3), keepdim=True)
        x_s_rect_miu = ((1 + x_s_rect_beta)) * x_s_disturb_miu
        x_s_rect_sigma = ((1 + x_s_rect_alpha)) * x_s_disturb_sigma
        x_s_rectify = ((x_s_disturb - x_s_disturb_miu) / (1e-6 + x_s_disturb_sigma)) * x_s_rect_sigma + x_s_rect_miu

        # Return the corrected feature map and its mean variance for loss analysis, La = La + Lc + L_BCE
        return x_q_rectify, x_s_rectify, x_q_rect_miu, x_q_rect_sigma, x_s_rect_miu, x_s_rect_sigma

    def cyclic_rectify(self, idx, alpha, beta, x_rect_miu, x_rect_sigma, feature):

        # Perform statistical correction and reverse calculate features
        second_dist_mean = (1 + beta) * x_rect_miu      # Perform mean alignment
        second_dist_sigma = (1 + alpha) * x_rect_sigma  # Perform variance alignment
        second_dist = ((feature - x_rect_miu) / (1e-6 + x_rect_sigma)) * second_dist_sigma + second_dist_mean

        # Correction of perturbed features through 7x7 dilated convolution
        second_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](second_dist.detach())
        second_rect_beta = second_rect.mean(dim=(2, 3), keepdim=True)  # Calculate the mean after convolution correction
        second_rect_alpha = second_rect.var(dim=(2, 3), keepdim=True)  # Calculate the variance after convolution correction

        second_rect_miu = (1 + second_rect_beta) * second_dist_mean
        second_rect_sigma = (1 + second_rect_alpha) * second_dist_sigma

        return second_rect_miu, second_rect_sigma  # Return the mean and variance after loop correction

    def masked_average_pooling(self, feature, mask):

        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)

        return masked_feature

    def similarity_func(self, feature_q, fg_proto, bg_proto):

        # Calculate cosine similarity with foreground prototype and background prototype respectively
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)  # similarity_fg.shape: [bsz, 50, 50]
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)  # similarity_bg.shape: [bsz, 50, 50]

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0

        return out

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)  # Reshape into the shape of (batch_2, 2, h * w)
        pred_fg = pred_1[:, 1]  # Foreground similarity
        pred_bg = pred_1[:, 0]  # Background similarity
        fg_ls = []              # Store significant foreground prototypes
        bg_ls = []              # Store significant background prototypes
        fg_local_ls = []        # Store self-support foreground prototypes
        bg_local_ls = []        # Store self-support foreground prototypes
        for epi in range(bs):
            fg_thres = 0.7  # foreground threshold t1
            bg_thres = 0.6  # background threshold t2
            cur_feat = feature_q[epi].view(1024, -1)  # Reshape the query features into the shape of (ch, h * w)
            f_h, f_w = feature_q[epi].shape[-2:]      # Obtain height and width

            # extract foreground pixels or extract 12 most likely pixels
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]  # shape: (ch, num_of_fg_pixels)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]

            # extract background pixels or extract 12 most likely pixels
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]  # shape: (ch, num_of_bg_pixels)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            # Get self-support foreground prototype and self-support background prototype
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)     # Normalized foreground features
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)     # Normalized background features
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)  # Normalize the original features
            cur_feat_norm_t = cur_feat_norm.t()  # transpose, from (ch, h*w) to (h*w, ch)
            # Calculate the similarity matrix A and normalize it
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0  # to obtain the foreground similarity matrix
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # to obtain the background similarity matrix
            fg_sim = fg_sim.softmax(-1)  # Perform softmax normalization
            bg_sim = bg_sim.softmax(-1)
            # Multiply the transposed foreground and background pixels to restore shape
            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)  # self-support foreground prototype
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)  # self-support background prototype
            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

            # Get significant foreground prototype and significant background prototype
            fg_proto = fg_feat.mean(-1)  # significant foreground prototype
            bg_proto = bg_feat.mean(-1)  # significant background prototype
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

        # significant prototype
        Pq_f = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        Pq_b = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # self-support prototype
        Pq_sspf = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        Pq_sspb = torch.cat(bg_local_ls, 0)

        return Pq_f, Pq_b, Pq_sspf, Pq_sspb

    def self_similarity(self, feature_q, fg_proto):

        # Calculate similarity with significant foreground prototypes to obtain a self-similar matrix
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)  # similarity_fg.shape: [bsz, 50, 50]

        # Judge foreground and background through threshold
        threshold = 0.5  # self-similar threshold t3
        foreground_mask = (similarity_fg >= threshold).float()
        M_self = foreground_mask

        return M_self

    def contribution_calculate(self, feature_q, proto1, proto2, proto3, proto4):

        # Calculate similarity
        mat1 = F.cosine_similarity(feature_q, proto1, dim=1)  # mat1.shape: [bsz, 50, 50]
        mat2 = F.cosine_similarity(feature_q, proto2, dim=1)  # mat2.shape: [bsz, 50, 50]
        mat3 = F.cosine_similarity(feature_q, proto3, dim=1)  # mat3.shape: [bsz, 50, 50]
        mat4 = F.cosine_similarity(feature_q, proto4, dim=1)  # mat4.shape: [bsz, 50, 50]

        max_f = torch.max(mat1, mat2)
        max_b = torch.max(mat3, mat4)

        fil_f = (max_f >= max_b).to(max_f.dtype)  # mat6.shape: [bsz, 50, 50]
        fil_b = (max_b >  max_f).to(max_b.dtype)  # mat6.shape: [bsz, 50, 50]

        # Generate compare matrix
        mat_5 = (mat1 >  mat2).to(mat1.dtype)  # mat5.shape: [bsz, 50, 50]
        mat_6 = (mat2 >= mat1).to(mat2.dtype)  # mat6.shape: [bsz, 50, 50]
        mat_7 = (mat3 >  mat4).to(mat3.dtype)  # mat5.shape: [bsz, 50, 50]
        mat_8 = (mat4 >= mat3).to(mat4.dtype)  # mat6.shape: [bsz, 50, 50]

        mat_cf1 = fil_f * mat_5
        mat_cf2 = fil_f * mat_6
        mat_cb1 = fil_b * mat_7
        mat_cb2 = fil_b * mat_8

        # to Compute contributions
        cf1 = mat_cf1.sum(dim=(1, 2))  # sum_mat5.shape: [bsz]
        cf2 = mat_cf2.sum(dim=(1, 2))
        cb1 = mat_cb1.sum(dim=(1, 2))
        cb2 = mat_cb2.sum(dim=(1, 2))

        # inner group contribution
        contr_f1 = (cf1 + 0.000001) / (cf1 + cf2 + 0.000002)
        contr_f2 = (cf2 + 0.000001) / (cf1 + cf2 + 0.000002)
        contr_b1 = (cb1 + 0.000001) / (cb1 + cb2 + 0.000002)
        contr_b2 = (cb2 + 0.000001) / (cb1 + cb2 + 0.000002)

        return contr_f1, contr_f2, contr_b1, contr_b2
