"""
평가 메트릭 모듈
- 문제 유형별 통계
- F1 Score, Accuracy 계산
- 평가 결과 출력
"""

from typing import List, Dict, Any
from collections import Counter


def compute_f1_score(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    예측값과 정답 레이블로 Macro F1 Score 및 Accuracy를 계산합니다.

    Args:
        predictions: 예측 답변 리스트 ("1", "2", "3", "4", "5")
        labels: 정답 레이블 리스트 ("1", "2", "3", "4", "5")

    Returns:
        {
            'accuracy': 정확도,
            'f1_macro': Macro F1 Score,
            'f1_weighted': Weighted F1 Score,
            'correct': 맞은 개수,
            'total': 전체 개수
        }
    """
    # 유효한 예측만 필터링 (answer가 "0"인 경우 파싱 실패)
    valid_pairs = [
        (pred, label)
        for pred, label in zip(predictions, labels)
        if pred != "0" and label is not None
    ]

    if not valid_pairs:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "correct": 0, "total": 0}

    valid_preds, valid_labels = zip(*valid_pairs)

    # Accuracy 계산
    correct = sum(1 for p, l in zip(valid_preds, valid_labels) if p == l)
    accuracy = correct / len(valid_pairs)

    # 모든 클래스 (1~5)
    all_classes = set(valid_preds) | set(valid_labels)

    # 클래스별 F1 계산
    f1_scores = []
    weighted_f1_scores = []
    label_counts = Counter(valid_labels)
    total_samples = len(valid_labels)

    for cls in all_classes:
        # True Positives, False Positives, False Negatives
        tp = sum(1 for p, l in zip(valid_preds, valid_labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(valid_preds, valid_labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(valid_preds, valid_labels) if p != cls and l == cls)

        # Precision, Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        # Weighted F1
        weight = label_counts[cls] / total_samples if cls in label_counts else 0
        weighted_f1_scores.append(f1 * weight)

    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    f1_weighted = sum(weighted_f1_scores)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "correct": correct,
        "total": len(valid_pairs),
    }


def print_evaluation_report(metrics_by_type: Dict[str, Dict[str, float]]) -> None:
    """
    평가 결과를 포맷팅하여 출력합니다.

    Args:
        metrics_by_type: compute_f1_by_question_type의 반환값
    """
    print("\n" + "=" * 60)
    print("                    평가 결과 (Evaluation Report)")
    # 전체 결과
    overall = metrics_by_type.get("overall", {})
    if overall.get("total", 0) > 0:
        print(
            f"  - Accuracy:     {overall['accuracy']*100:.2f}% ({overall['correct']}/{overall['total']})"
        )
        print(f"  - F1 (Macro):   {overall['f1_macro']*100:.2f}%")
        print(f"  - F1 (Weighted): {overall['f1_weighted']*100:.2f}%")

    print("=" * 60 + "\n")
