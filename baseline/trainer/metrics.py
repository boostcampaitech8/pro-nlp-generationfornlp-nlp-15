# src/metrics.py
import torch
import numpy as np
import evaluate
from sklearn.metrics import f1_score
from common.utils.template import get_response_template


class CustomMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.acc_metric = evaluate.load("accuracy")
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        # 정답 토큰("1"~"5")의 ID를 미리 찾아둡니다.
        self.logit_idx = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

        # response_template 토큰 ID (정답 직전 위치 찾기용)
        # 자동 감지된 템플릿 사용
        response_template = get_response_template(self.tokenizer)
        print(
            f"[CustomMetrics] Auto-detected response_template: {repr(response_template)}"
        )

        self.response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )

        self._debug_printed = False  # 디버깅 로그 1회만 출력

    def _find_response_start(self, input_ids) -> int:
        """
        input_ids(또는 labels)에서 정답 시작 위치를 찾습니다.

        completion_only_loss=True 환경에서는 labels의 prompt 부분이 -100으로
        마스킹되므로, -100이 아닌 첫 번째 토큰 위치를 정답 시작 위치로 반환합니다.

        Returns:
            정답 시작 위치 (0-indexed). 못 찾은 경우 -1.
        """
        if isinstance(input_ids, torch.Tensor):
            input_list = input_ids.tolist()
        else:
            input_list = list(input_ids)

        # 방법 1: -100이 아닌 첫 번째 토큰 위치 찾기 (completion_only_loss=True 환경)
        # labels가 마스킹된 경우: prompt 부분은 -100, completion 부분만 실제 토큰 ID
        for i, token_id in enumerate(input_list):
            if token_id != -100:
                return i  # 첫 번째 유효한 토큰 = 정답 시작 위치

        # 방법 2 (fallback): response_template 패턴 탐색 (마스킹되지 않은 경우)
        template_len = len(self.response_template_ids)
        for i in range(len(input_list) - template_len, -1, -1):
            if input_list[i : i + template_len] == self.response_template_ids:
                return i + template_len

        return -1  # 못 찾은 경우

    def _find_answer_token_position(self, label_seq) -> int:
        """
        CoT 형식에서 '정답:' 다음 토큰(정답 숫자)의 위치를 찾습니다.
        
        labels에서 -100이 아닌 토큰을 모두 디코딩하여 '정답:' 패턴을 찾고,
        그 위치에 해당하는 토큰 인덱스를 반환합니다.
        
        Returns:
            정답 토큰 위치 (0-indexed). 못 찾은 경우 -1.
        """
        if hasattr(label_seq, 'tolist'):
            label_list = label_seq.tolist()
        else:
            label_list = list(label_seq)
        
        # 유효한 토큰과 원본 인덱스 매핑
        valid_positions = []
        valid_token_ids = []
        for i, t in enumerate(label_list):
            if t != -100 and t >= 0:
                valid_positions.append(i)
                valid_token_ids.append(t)
        
        if not valid_token_ids:
            return -1
        
        # '정답' 관련 토큰 ID 찾기
        # 토크나이저마다 다를 수 있으므로, 각 토큰을 개별 디코딩하여 패턴 탐색
        try:
            # 역으로 탐색 (마지막 '정답:' 사용)
            for i in range(len(valid_token_ids) - 1, -1, -1):
                decoded = self.tokenizer.decode([valid_token_ids[i]], skip_special_tokens=True)
                if decoded.strip() in ['1', '2', '3', '4', '5']:
                    # 이 토큰이 정답 숫자일 수 있음, 앞에 '정답' 또는 ':' 있는지 확인
                    if i >= 1:
                        prev_decoded = self.tokenizer.decode(valid_token_ids[max(0, i-3):i], skip_special_tokens=True)
                        if '정답' in prev_decoded or ':' in prev_decoded:
                            return valid_positions[i]
            
            # Fallback: 마지막 숫자 토큰 찾기
            for i in range(len(valid_token_ids) - 1, -1, -1):
                decoded = self.tokenizer.decode([valid_token_ids[i]], skip_special_tokens=True).strip()
                if decoded in ['1', '2', '3', '4', '5']:
                    return valid_positions[i]
        except:
            pass
        
        return -1
    
    def preprocess_logits_for_metrics(self, logits, labels):
        """
        모델의 Logits 중 정답 토큰(1~5)에 해당하는 부분만 추출합니다.

        CoT 형식에서는 "정답: X" 패턴에서 X 토큰을 예측해야 합니다.
        따라서 "정답:" 토큰 직전 위치에서의 logits를 사용합니다.
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]

        batch_size = logits.size(0)
        batch_logits = []

        for i in range(batch_size):
            # CoT 형식: 정답 토큰 위치 찾기
            answer_pos = self._find_answer_token_position(labels[i])

            if answer_pos > 0:
                # answer_pos - 1 위치에서 정답 토큰 예측
                pred_pos = answer_pos - 1
                batch_logits.append(logits[i, pred_pos, self.logit_idx])
            else:
                # fallback: 마지막에서 3번째 위치 사용
                batch_logits.append(logits[i, -3, self.logit_idx])

        return torch.stack(batch_logits)

    def compute_metrics(self, evaluation_result):
        """
        평가 결과에서 macro_f1과 accuracy를 계산합니다.

        CoT 형식에서는 completion이 다음과 같은 형태입니다:
        "**관련 지식...**\n\n정답: 2<end_of_turn>"
        
        따라서 labels를 디코딩하여 "정답: X" 패턴을 찾아야 합니다.
        """
        import re
        
        logits, labels = evaluation_result

        # 디버깅: 첫 번째 샘플의 labels 정보 출력 (1회만)
        if not self._debug_printed and len(labels) > 0:
            self._debug_printed = True
            
        # 1. 라벨에서 정답 추출 ("정답: X" 패턴 파싱)
        parsed_labels = []
        valid_indices = []
        
        # 정답 패턴: "정답: 1", "정답:2", "정답 : 3" 등
        answer_pattern = re.compile(r'정답\s*:\s*([1-5])')

        for idx, label_seq in enumerate(labels):
            # -100이 아닌 토큰만 추출
            if hasattr(label_seq, 'tolist'):
                label_list = label_seq.tolist()
            else:
                label_list = list(label_seq)
            
            # 유효한 토큰 ID만 추출 (-100 제외)
            valid_token_ids = [t for t in label_list if t != -100 and t >= 0]
            
            if not valid_token_ids:
                continue
                
            # 디코딩
            try:
                decoded_text = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)
            except:
                continue
            
            # 정답 패턴 찾기 (마지막 매칭 사용)
            matches = answer_pattern.findall(decoded_text)
            if matches:
                answer_str = matches[-1]  # 마지막 "정답: X" 사용
                if answer_str in self.int_output_map:
                    parsed_labels.append(self.int_output_map[answer_str])
                    valid_indices.append(idx)
            else:
                # Fallback: CoT 형식이 아닌 경우 (Standard SFT: "1")
                # 텍스트에서 1~5 숫자 찾기
                simple_digit = re.findall(r'([1-5])', decoded_text)
                if simple_digit:
                    answer_str = simple_digit[-1]
                    if answer_str in self.int_output_map:
                        parsed_labels.append(self.int_output_map[answer_str])
                        valid_indices.append(idx)

        # 디버깅: 결과 출력
        print(
            f"[DEBUG] valid_indices count: {len(valid_indices)}, parsed_labels sample: {parsed_labels[:5] if parsed_labels else 'empty'}"
        )

        if not valid_indices:
            return {
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "valid_sample_count": 0,
            }

        # 2. 예측값 추출 (Argmax) - 유효한 샘플만
        # torch tensor를 numpy로 변환
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = np.array(logits)

        probs = torch.nn.functional.softmax(torch.tensor(logits_np), dim=-1).numpy()
        predictions = np.argmax(probs, axis=-1)
        filtered_preds = np.array([predictions[i] for i in valid_indices])
        filtered_labels = np.array(parsed_labels)

        # 3. Macro F1-score 및 정확도 계산
        macro_f1 = f1_score(
            filtered_labels, filtered_preds, average="macro", zero_division=0
        )
        acc = float(np.mean(filtered_labels == filtered_preds))

        return {
            "macro_f1": float(macro_f1),
            "accuracy": acc,
            "valid_sample_count": len(valid_indices),
        }
