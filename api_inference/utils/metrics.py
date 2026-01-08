"""
평가 메트릭 모듈
- 문제 유형별 통계
- F1 Score, Accuracy 계산
- 평가 결과 출력
"""
from typing import List, Dict, Any
from collections import Counter

from api_inference.prompts import QuestionType


def get_question_type_stats(test_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    테스트 데이터의 문제 유형별 통계를 반환합니다.
    
    Args:
        test_data: load_test_data로 로드한 테스트 데이터
    
    Returns:
        유형별 문제 개수 딕셔너리
    """
    stats = {qt.value: 0 for qt in QuestionType}
    for item in test_data:
        question_type = item.get('question_type', QuestionType.DEFAULT)
        stats[question_type.value] += 1
    return stats


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
        (pred, label) for pred, label in zip(predictions, labels)
        if pred != "0" and label is not None
    ]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'correct': 0,
            'total': 0
        }
    
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
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'correct': correct,
        'total': len(valid_pairs)
    }


def compute_f1_by_question_type(
    results: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    문제 유형별 F1 Score를 계산합니다.
    
    Args:
        results: API 추론 결과 리스트 [{'id': ..., 'answer': ..., 'question_type': ...}, ...]
        test_data: 정답이 포함된 테스트 데이터 리스트
    
    Returns:
        유형별 metrics 딕셔너리
        {
            'multi_label': {'accuracy': ..., 'f1_macro': ..., ...},
            'single_correct': {...},
            ...
            'overall': {...}
        }
    """
    # ID로 정답 매핑
    id_to_answer = {item['id']: item.get('answer') for item in test_data}
    id_to_type = {item['id']: item.get('question_type', QuestionType.DEFAULT) for item in test_data}
    
    # 유형별로 그룹화
    type_predictions = {qt.value: [] for qt in QuestionType}
    type_labels = {qt.value: [] for qt in QuestionType}
    
    all_predictions = []
    all_labels = []
    
    for result in results:
        item_id = result['id']
        pred = result['answer']
        label = id_to_answer.get(item_id)
        q_type = id_to_type.get(item_id, QuestionType.DEFAULT)
        
        if isinstance(q_type, QuestionType):
            q_type = q_type.value
        
        if label is not None:
            type_predictions[q_type].append(pred)
            type_labels[q_type].append(label)
            all_predictions.append(pred)
            all_labels.append(label)
    
    # 유형별 metrics 계산
    metrics_by_type = {}
    for q_type in QuestionType:
        preds = type_predictions[q_type.value]
        labels = type_labels[q_type.value]
        if preds and labels:
            metrics_by_type[q_type.value] = compute_f1_score(preds, labels)
        else:
            metrics_by_type[q_type.value] = {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'correct': 0,
                'total': 0
            }
    
    # 전체 metrics
    metrics_by_type['overall'] = compute_f1_score(all_predictions, all_labels)
    
    return metrics_by_type


def print_evaluation_report(metrics_by_type: Dict[str, Dict[str, float]]) -> None:
    """
    평가 결과를 포맷팅하여 출력합니다.
    
    Args:
        metrics_by_type: compute_f1_by_question_type의 반환값
    """
    print("\n" + "=" * 60)
    print("                    평가 결과 (Evaluation Report)")
    print("=" * 60)
    
    # 유형별 결과
    print("\n[문제 유형별 성능]")
    print("-" * 60)
    print(f"{'유형':<20} {'정확도':>10} {'F1(Macro)':>12} {'정답':>10}")
    print("-" * 60)
    
    for q_type in QuestionType:
        metrics = metrics_by_type.get(q_type.value, {})
        if metrics.get('total', 0) > 0:
            print(f"{q_type.value:<20} {metrics['accuracy']*100:>9.2f}% {metrics['f1_macro']*100:>11.2f}% {metrics['correct']:>5}/{metrics['total']:<4}")
    
    print("-" * 60)
    
    # 전체 결과
    overall = metrics_by_type.get('overall', {})
    if overall.get('total', 0) > 0:
        print(f"\n[전체 성능 (Overall)]")
        print(f"  - Accuracy:     {overall['accuracy']*100:.2f}% ({overall['correct']}/{overall['total']})")
        print(f"  - F1 (Macro):   {overall['f1_macro']*100:.2f}%")
        print(f"  - F1 (Weighted): {overall['f1_weighted']*100:.2f}%")
    
    print("=" * 60 + "\n")
