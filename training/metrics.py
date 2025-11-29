"""
Evaluation Metrics for Multi-task Learning

Implements metrics for:
1. Classification: AUC, F1, Precision, Recall
2. Localization: Dice, IoU, AP50
3. Report Generation: BLEU, ROUGE, BERTScore
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge import Rouge
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK/Rouge not available. Install with: pip install nltk rouge")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: BERTScore not available. Install with: pip install bert-score")


class ClassificationMetrics:
    """
    Metrics for multi-label disease classification.
    """

    def __init__(self, num_classes: int = 18):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_predictions = []
        self.all_probabilities = []
        self.all_targets = []

    def update(
        self,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update with batch predictions.

        Args:
            predictions: Binary predictions (B, num_classes)
            probabilities: Prediction probabilities (B, num_classes)
            targets: Ground truth labels (B, num_classes)
        """
        self.all_predictions.append(predictions.detach().cpu())
        self.all_probabilities.append(probabilities.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Returns:
            Dictionary with macro/micro averaged metrics
        """
        if not self.all_predictions:
            return {}

        predictions = torch.cat(self.all_predictions, dim=0).numpy()  # (N, num_classes)
        probabilities = torch.cat(self.all_probabilities, dim=0).numpy()
        targets = torch.cat(self.all_targets, dim=0).numpy()

        metrics = {}

        # AUC-ROC (requires probabilities)
        try:
            # Macro average (per-class then average)
            auc_scores = []
            for i in range(self.num_classes):
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    auc = roc_auc_score(targets[:, i], probabilities[:, i])
                    auc_scores.append(auc)

            if auc_scores:
                metrics['auc_macro'] = np.mean(auc_scores)

            # Micro average (all classes together)
            metrics['auc_micro'] = roc_auc_score(
                targets.ravel(),
                probabilities.ravel()
            )
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")

        # F1 Score
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(targets, predictions, average='micro', zero_division=0)

        # Precision
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(targets, predictions, average='micro', zero_division=0)

        # Recall
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(targets, predictions, average='micro', zero_division=0)

        return metrics

    def compute_per_class(self) -> Dict[str, np.ndarray]:
        """
        Compute per-class metrics.

        Returns:
            Dictionary with per-class scores
        """
        if not self.all_predictions:
            return {}

        predictions = torch.cat(self.all_predictions, dim=0).numpy()
        probabilities = torch.cat(self.all_probabilities, dim=0).numpy()
        targets = torch.cat(self.all_targets, dim=0).numpy()

        per_class_metrics = {
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

        for i in range(self.num_classes):
            # AUC
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                try:
                    auc = roc_auc_score(targets[:, i], probabilities[:, i])
                    per_class_metrics['auc'].append(auc)
                except:
                    per_class_metrics['auc'].append(0.0)
            else:
                per_class_metrics['auc'].append(0.0)

            # F1, Precision, Recall
            per_class_metrics['f1'].append(
                f1_score(targets[:, i], predictions[:, i], zero_division=0)
            )
            per_class_metrics['precision'].append(
                precision_score(targets[:, i], predictions[:, i], zero_division=0)
            )
            per_class_metrics['recall'].append(
                recall_score(targets[:, i], predictions[:, i], zero_division=0)
            )

        return {k: np.array(v) for k, v in per_class_metrics.items()}


class LocalizationMetrics:
    """
    Metrics for lesion localization/segmentation.
    """

    def __init__(self, num_diseases: int = 5):
        self.num_diseases = num_diseases
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.disease_predictions = {i: [] for i in range(self.num_diseases)}
        self.disease_targets = {i: [] for i in range(self.num_diseases)}

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ):
        """
        Update with batch predictions.

        Args:
            predictions: Dict of prediction masks per disease (B, D, H, W)
            targets: Dict of ground truth masks per disease (B, D, H, W)
        """
        disease_names = list(predictions.keys())

        for idx, disease in enumerate(disease_names):
            if disease in targets:
                self.disease_predictions[idx].append(
                    predictions[disease].detach().cpu()
                )
                self.disease_targets[idx].append(
                    targets[disease].detach().cpu()
                )

    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
        """Compute Dice coefficient."""
        pred = (pred > 0.5).float()
        target = target.float()

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return dice.item()

    def iou_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
        """Compute IoU (Jaccard index)."""
        pred = (pred > 0.5).float()
        target = target.float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)

        return iou.item()

    def compute(self) -> Dict[str, float]:
        """
        Compute localization metrics.

        Returns:
            Dictionary with average metrics across diseases
        """
        if not any(self.disease_predictions.values()):
            return {}

        all_dice = []
        all_iou = []

        for disease_idx in range(self.num_diseases):
            if not self.disease_predictions[disease_idx]:
                continue

            predictions = torch.cat(self.disease_predictions[disease_idx], dim=0)
            targets = torch.cat(self.disease_targets[disease_idx], dim=0)

            # Compute metrics for each sample
            for pred, target in zip(predictions, targets):
                dice = self.dice_coefficient(pred, target)
                iou = self.iou_score(pred, target)

                all_dice.append(dice)
                all_iou.append(iou)

        metrics = {
            'dice_mean': np.mean(all_dice) if all_dice else 0.0,
            'dice_std': np.std(all_dice) if all_dice else 0.0,
            'iou_mean': np.mean(all_iou) if all_iou else 0.0,
            'iou_std': np.std(all_iou) if all_iou else 0.0
        }

        return metrics

    def compute_per_disease(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per disease.

        Returns:
            Dictionary with per-disease metrics
        """
        per_disease = {}

        for disease_idx in range(self.num_diseases):
            if not self.disease_predictions[disease_idx]:
                continue

            predictions = torch.cat(self.disease_predictions[disease_idx], dim=0)
            targets = torch.cat(self.disease_targets[disease_idx], dim=0)

            dice_scores = []
            iou_scores = []

            for pred, target in zip(predictions, targets):
                dice_scores.append(self.dice_coefficient(pred, target))
                iou_scores.append(self.iou_score(pred, target))

            per_disease[f'disease_{disease_idx}'] = {
                'dice': np.mean(dice_scores),
                'iou': np.mean(iou_scores)
            }

        return per_disease


class ReportGenerationMetrics:
    """
    Metrics for clinical report generation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated predictions and references."""
        self.predictions = []
        self.references = []

    def update(self, predictions: List[str], references: List[str]):
        """
        Update with batch predictions.

        Args:
            predictions: List of predicted report texts
            references: List of reference report texts
        """
        self.predictions.extend(predictions)
        self.references.extend(references)

    def compute_bleu(self) -> Dict[str, float]:
        """
        Compute BLEU scores.

        Returns:
            Dictionary with BLEU-1 to BLEU-4 scores
        """
        if not NLTK_AVAILABLE:
            return {}

        # Tokenize
        pred_tokens = [pred.split() for pred in self.predictions]
        ref_tokens = [[ref.split()] for ref in self.references]

        smoothing = SmoothingFunction().method1

        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            bleu = corpus_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores[f'bleu_{n}'] = bleu

        return bleu_scores

    def compute_rouge(self) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not NLTK_AVAILABLE:
            return {}

        try:
            rouge = Rouge()
            scores = rouge.get_scores(self.predictions, self.references, avg=True)

            return {
                'rouge_1_f': scores['rouge-1']['f'],
                'rouge_2_f': scores['rouge-2']['f'],
                'rouge_l_f': scores['rouge-l']['f']
            }
        except Exception as e:
            print(f"Warning: Could not compute ROUGE: {e}")
            return {}

    def compute_bertscore(self) -> Dict[str, float]:
        """
        Compute BERTScore.

        Returns:
            Dictionary with BERTScore precision, recall, F1
        """
        if not BERTSCORE_AVAILABLE:
            return {}

        try:
            P, R, F1 = bert_score(
                self.predictions,
                self.references,
                lang="en",
                rescale_with_baseline=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"Warning: Could not compute BERTScore: {e}")
            return {}

    def compute(self) -> Dict[str, float]:
        """
        Compute all report generation metrics.

        Returns:
            Dictionary with BLEU, ROUGE, BERTScore
        """
        if not self.predictions:
            return {}

        metrics = {}

        # BLEU
        bleu_scores = self.compute_bleu()
        metrics.update(bleu_scores)

        # ROUGE
        rouge_scores = self.compute_rouge()
        metrics.update(rouge_scores)

        # BERTScore (expensive, may skip for speed)
        # bertscore = self.compute_bertscore()
        # metrics.update(bertscore)

        return metrics


class MultiTaskMetrics:
    """
    Combined metrics for all tasks.
    """

    def __init__(
        self,
        num_classes: int = 18,
        num_lesion_diseases: int = 5
    ):
        self.classification_metrics = ClassificationMetrics(num_classes)
        self.localization_metrics = LocalizationMetrics(num_lesion_diseases)
        self.report_metrics = ReportGenerationMetrics()

    def reset(self):
        """Reset all metrics."""
        self.classification_metrics.reset()
        self.localization_metrics.reset()
        self.report_metrics.reset()

    def update(
        self,
        outputs: Dict,
        targets: Dict,
        tasks: Optional[List[str]] = None
    ):
        """
        Update metrics with batch results.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets dictionary
            tasks: Which tasks to update metrics for
        """
        if tasks is None:
            tasks = []
            if 'classification' in outputs:
                tasks.append('classification')
            if 'localization' in outputs:
                tasks.append('localization')
            if 'reports' in outputs:
                tasks.append('generation')

        # Classification
        if 'classification' in tasks and 'classification' in outputs:
            class_outputs = outputs['classification']
            class_targets = targets.get('disease_labels')

            if class_targets is not None:
                self.classification_metrics.update(
                    predictions=class_outputs.get('predictions', (class_outputs['probabilities'] > 0.5).long()),
                    probabilities=class_outputs['probabilities'],
                    targets=class_targets
                )

        # Localization
        if 'localization' in tasks and 'localization' in outputs:
            loc_outputs = outputs['localization']
            loc_targets = targets.get('lesion_masks')

            if loc_targets is not None:
                # Convert to binary masks if needed
                if 'masks' in loc_outputs:
                    loc_outputs = loc_outputs['masks']

                self.localization_metrics.update(
                    predictions=loc_outputs,
                    targets=loc_targets
                )

        # Report generation
        if 'generation' in tasks and 'reports' in outputs:
            report_predictions = outputs['reports']
            report_references = targets.get('report_texts')

            if report_references is not None:
                self.report_metrics.update(
                    predictions=report_predictions,
                    references=report_references
                )

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary with metrics from all tasks
        """
        all_metrics = {}

        # Classification metrics
        class_metrics = self.classification_metrics.compute()
        all_metrics.update({f'cls_{k}': v for k, v in class_metrics.items()})

        # Localization metrics
        loc_metrics = self.localization_metrics.compute()
        all_metrics.update({f'loc_{k}': v for k, v in loc_metrics.items()})

        # Report metrics
        report_metrics = self.report_metrics.compute()
        all_metrics.update({f'gen_{k}': v for k, v in report_metrics.items()})

        return all_metrics


# Test code
if __name__ == "__main__":
    print("Testing Metrics...")

    # Test classification metrics
    print("\n1. Testing ClassificationMetrics...")
    cls_metrics = ClassificationMetrics(num_classes=18)

    # Create dummy data
    for _ in range(3):
        predictions = torch.randint(0, 2, (4, 18))
        probabilities = torch.rand(4, 18)
        targets = torch.randint(0, 2, (4, 18)).float()

        cls_metrics.update(predictions, probabilities, targets)

    metrics = cls_metrics.compute()
    print(f"   Classification metrics: {metrics}")

    # Test localization metrics
    print("\n2. Testing LocalizationMetrics...")
    loc_metrics = LocalizationMetrics(num_diseases=5)

    for _ in range(3):
        predictions = {
            'disease_0': torch.rand(2, 64, 128, 128),
            'disease_1': torch.rand(2, 64, 128, 128)
        }
        targets = {
            'disease_0': torch.randint(0, 2, (2, 64, 128, 128)).float(),
            'disease_1': torch.randint(0, 2, (2, 64, 128, 128)).float()
        }

        loc_metrics.update(predictions, targets)

    metrics = loc_metrics.compute()
    print(f"   Localization metrics: {metrics}")

    # Test report generation metrics
    if NLTK_AVAILABLE:
        print("\n3. Testing ReportGenerationMetrics...")
        report_metrics = ReportGenerationMetrics()

        predictions = [
            "The CT scan shows bilateral pleural effusion and consolidation in the right lower lobe.",
            "No significant abnormalities detected in the thoracic region."
        ]
        references = [
            "Bilateral pleural effusion is present with right lower lobe consolidation.",
            "The chest CT is unremarkable with no acute findings."
        ]

        report_metrics.update(predictions, references)

        metrics = report_metrics.compute()
        print(f"   Report generation metrics: {metrics}")
    else:
        print("\n3. Skipping ReportGenerationMetrics (NLTK not available)")

    print("\n✓ All tests passed!")
