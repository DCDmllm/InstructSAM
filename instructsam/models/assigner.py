from typing import List, Union, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DiceCost:
    """Cost of mask assignments based on dice losses.

    Args:
        pred_act (bool): Whether to apply sigmoid to mask_pred. Defaults to True.
        eps (float): Defaults to 1e-3.
        naive_dice (bool): Whether to use naive dice loss. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, pred_act: bool = True, eps: float = 1e-3, naive_dice: bool = True, weight: Union[float, int] = 0.5) -> None:
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice
        self.weight = weight

    def _binary_mask_dice_loss(self, mask_preds: Tensor, gt_masks: Tensor) -> Tensor:
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_queries, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *).
        Returns:
            Tensor: Dice cost matrix in shape (num_queries, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()

        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, pred_masks, gt_masks, pred_logits) -> Tensor:
        """Compute match cost."""
        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


class CrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid of softmax. Defaults to True.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, use_sigmoid: bool = True, weight: Union[float, int] = 2.) -> None:
        self.use_sigmoid = use_sigmoid
        self.weight = weight

    def _binary_cross_entropy(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Prediction tensor.
            gt_labels (Tensor): Ground truth tensor.
        Returns:
            Tensor: Cost matrix.
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()

        pos = F.binary_cross_entropy_with_logits(cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(cls_pred, torch.zeros_like(cls_pred), reduction='none')

        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost /= cls_pred.shape[1]
        return cls_cost

    def __call__(self, pred_masks, gt_masks, pred_logits) -> Tensor:
        """Compute match cost."""
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError
        return cls_cost * self.weight

class ClassificationCost:
    """ClsCost.
    """
    def __init__(self, weight: Union[float, int] = 0.5):
        self.weight = weight

    def __call__(self, pred_masks, gt_masks, pred_logits) -> Tensor:
        # pred_logits: (num_queries,)
        cost_pos = F.softplus(-pred_logits)  # (Q,)
        cls_cost = cost_pos[:, None].expand(-1, gt_masks.shape[0])
        return cls_cost * self.weight


class HungarianAssigner():
    """Computes one-to-one matching between predictions and ground truth."""

    def __init__(self, dice_loss_weight, ce_loss_weight, cls_loss_weight) -> None:
        self.dice_cost = DiceCost(weight=dice_loss_weight)
        self.ce_cost = CrossEntropyLossCost(weight=ce_loss_weight)
        self.cls_cost = ClassificationCost(weight=cls_loss_weight)
        self.match_costs = [self.dice_cost, self.ce_cost, self.cls_cost]
        # self.match_costs = [self.dice_cost, self.ce_cost]

    def assign(self, pred_masks, gt_masks, pred_logits):
        num_gts, num_preds = len(gt_masks), len(pred_masks)
        device = pred_masks.device

        # Default assignment
        assigned_gt_inds = torch.full((num_preds,), -1, dtype=torch.long, device=device)

        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = -1
            return assigned_gt_inds

        # Compute weighted costs
        cost_list = [
          cost(pred_masks, gt_masks, pred_logits) for cost in self.match_costs
        ]
        cost = torch.stack(cost_list).sum(dim=0)

        # Hungarian matching
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

        matched_row_inds = torch.tensor(matched_row_inds, device=device)
        matched_col_inds = torch.tensor(matched_col_inds, device=device)

        # Assign results
        assigned_gt_inds[:] = -1
        assigned_gt_inds[matched_row_inds] = matched_col_inds

        return assigned_gt_inds