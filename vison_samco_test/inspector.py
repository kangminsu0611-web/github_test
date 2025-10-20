#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공용 검사 모듈: MissingPartInspector

정상 이미지에서 ROI 템플릿을 저장하고, 검사 이미지에서 동일 위치의 패턴 유사도를 계산해
OK/NG 판정을 내리는 핵심 로직을 제공합니다.

명렁어 : .\venv\Scripts\python.exe .\run_inspection.py --method combined --threshold 0.5

"""

from typing import List, Tuple, Dict
import numpy as np
import cv2
import os
import json


class MissingPartInspector:
    def __init__(self, threshold: float = 0.8, normalize_size: bool = True, target_size: tuple | None = None):
        self.threshold = threshold
        self.normalize_size = normalize_size
        self.target_size = target_size
        self.templates: Dict[int, np.ndarray] = {}
        self.roi_coords: List[Tuple[int, int, int, int]] = []
        self.original_roi_coords: List[Tuple[int, int, int, int]] = []

    def _normalize_image(self, img: np.ndarray, target_size: tuple | None) -> tuple[np.ndarray, tuple, tuple]:
        if not self.normalize_size:
            h, w = img.shape[:2]
            return img, (h, w), (1.0, 1.0)

        h, w = img.shape[:2]
        if target_size is None:
            if w > h:
                target_w = min(640, w)
                target_h = int(h * target_w / w)
            else:
                target_h = min(640, h)
                target_w = int(w * target_h / h)
            target_size = (target_w, target_h)

        target_w, target_h = target_size
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        if new_w == target_w and new_h == target_h:
            return resized, (h, w), (scale, scale)

        padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return padded, (h, w), (scale, scale)

    def _denormalize_coords(self, coords: List[Tuple[int, int, int, int]], original_size: tuple, scale: tuple) -> List[Tuple[int, int, int, int]]:
        scale_x, scale_y = scale
        orig_h, orig_w = original_size
        denorm: List[Tuple[int, int, int, int]] = []
        for x, y, w, h in coords:
            dx = int(x / scale_x)
            dy = int(y / scale_y)
            dw = int(w / scale_x)
            dh = int(h / scale_y)
            dx = max(0, min(dx, orig_w - dw))
            dy = max(0, min(dy, orig_h - dh))
            denorm.append((dx, dy, dw, dh))
        return denorm

    def select_rois(self, image_path: str, display_scale: float = 1.2) -> List[Tuple[int, int, int, int]]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        if self.normalize_size:
            norm_img, original_size, norm_scale = self._normalize_image(img, self.target_size)
        else:
            norm_img, original_size, norm_scale = img, img.shape[:2], (1.0, 1.0)

        h, w = norm_img.shape[:2]
        vw = int(w * display_scale)
        vh = int(h * display_scale)
        vis = cv2.resize(norm_img, (vw, vh))

        print("ROI를 선택하세요 (ESC로 종료, 다중 선택 가능)")
        rois = cv2.selectROIs("ROI 선택", vis, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if len(rois) == 0:
            print("선택된 ROI가 없습니다.")
            return []

        normalized_rois: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in rois:
            nx = int(x / display_scale)
            ny = int(y / display_scale)
            nw = int(w / display_scale)
            nh = int(h / display_scale)
            normalized_rois.append((nx, ny, nw, nh))

        if self.normalize_size:
            self.original_roi_coords = self._denormalize_coords(normalized_rois, original_size, norm_scale)
            return self.original_roi_coords
        return normalized_rois

    def save_templates(self, normal_image_path: str, roi_coords: List[Tuple[int, int, int, int]], save_dir: str = "templates") -> None:
        if not os.path.exists(normal_image_path):
            raise FileNotFoundError(f"정상 이미지 파일을 찾을 수 없습니다: {normal_image_path}")

        img = cv2.imread(normal_image_path)
        if img is None:
            raise ValueError(f"정상 이미지를 읽을 수 없습니다: {normal_image_path}")

        if self.normalize_size:
            norm_img, original_size, norm_scale = self._normalize_image(img, self.target_size)
        else:
            norm_img, original_size, norm_scale = img, img.shape[:2], (1.0, 1.0)

        os.makedirs(save_dir, exist_ok=True)

        normalized_roi_coords: List[Tuple[int, int, int, int]] = []
        for i, (x, y, w, h) in enumerate(roi_coords):
            if self.normalize_size:
                nx = int(x * norm_scale[0])
                ny = int(y * norm_scale[1])
                nw = int(w * norm_scale[0])
                nh = int(h * norm_scale[1])
                normalized_roi_coords.append((nx, ny, nw, nh))
                roi = norm_img[ny:ny + nh, nx:nx + nw]
            else:
                roi = img[y:y + h, x:x + w]
                normalized_roi_coords.append((x, y, w, h))

            cv2.imwrite(os.path.join(save_dir, f"template_roi{i + 1}.jpg"), roi)

        coords_data = {
            "original_coords": roi_coords,
            "normalized_coords": normalized_roi_coords,
            "normalize_size": self.normalize_size,
            "target_size": self.target_size,
            "original_image_size": original_size,
            "normalized_image_size": norm_img.shape[:2],
        }
        with open(os.path.join(save_dir, "roi_coords.json"), "w", encoding="utf-8") as f:
            json.dump(coords_data, f, indent=2, ensure_ascii=False)

        self.roi_coords = normalized_roi_coords

    def load_templates(self, save_dir: str = "templates") -> None:
        coords_path = os.path.join(save_dir, "roi_coords.json")
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"ROI 좌표 파일을 찾을 수 없습니다: {coords_path}")

        with open(coords_path, "r", encoding="utf-8") as f:
            coords_data = json.load(f)

        if isinstance(coords_data, dict) and "normalized_coords" in coords_data:
            self.roi_coords = coords_data["normalized_coords"]
            self.original_roi_coords = coords_data.get("original_coords", [])
            self.normalize_size = coords_data.get("normalize_size", True)
            self.target_size = tuple(coords_data.get("target_size")) if coords_data.get("target_size") else None
        else:
            self.roi_coords = coords_data
            self.original_roi_coords = coords_data
            self.normalize_size = False
            self.target_size = None

        self.templates = {}
        for i in range(len(self.roi_coords)):
            path = os.path.join(save_dir, f"template_roi{i + 1}.jpg")
            if os.path.exists(path):
                self.templates[i + 1] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _similarity_orb(template: np.ndarray, test_roi: np.ndarray) -> float:
        """개선된 ORB 유사도 계산"""
        try:
            # 크기 맞추기
            if template.shape != test_roi.shape:
                test_roi = cv2.resize(test_roi, (template.shape[1], template.shape[0]))
            
            # 여러 ORB 설정 시도
            orb_configs = [
                {"nfeatures": 1000, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 15},
                {"nfeatures": 500, "scaleFactor": 1.1, "nlevels": 6, "edgeThreshold": 10},
                {"nfeatures": 2000, "scaleFactor": 1.3, "nlevels": 10, "edgeThreshold": 20}
            ]
            
            best_score = 0.0
            
            for config in orb_configs:
                try:
                    orb = cv2.ORB_create(**config)
                    kp1, des1 = orb.detectAndCompute(template, None)
                    kp2, des2 = orb.detectAndCompute(test_roi, None)
                    
                    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
                        continue
                    
                    # BF 매처 사용 (더 안정적)
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    if not matches:
                        continue
                    
                    # 매칭 품질 계산
                    distances = [m.distance for m in matches]
                    avg_distance = float(np.mean(distances))
                    max_distance = float(np.max(distances))
                    
                    # 거리 기반 점수 (낮을수록 좋음)
                    distance_score = max(0.0, 1.0 - (avg_distance / 100.0))
                    
                    # 매칭 비율 점수
                    match_ratio = len(matches) / min(len(des1), len(des2))
                    
                    # 일관성 점수 (거리 분산이 낮을수록 좋음)
                    distance_std = float(np.std(distances))
                    consistency_score = max(0.0, 1.0 - (distance_std / 50.0))
                    
                    # 최종 점수 계산
                    score = 0.4 * distance_score + 0.3 * match_ratio + 0.3 * consistency_score
                    best_score = max(best_score, score)
                    
                except Exception:
                    continue
            
            # SIFT도 시도 (ORB가 실패한 경우)
            if best_score < 0.3:
                try:
                    sift = cv2.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(template, None)
                    kp2, des2 = sift.detectAndCompute(test_roi, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 5 and len(des2) > 5:
                        bf = cv2.BFMatcher()
                        matches = bf.knnMatch(des1, des2, k=2)
                        
                        # Lowe's ratio test
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.75 * n.distance:
                                    good_matches.append(m)
                        
                        if len(good_matches) > 4:
                            match_ratio = len(good_matches) / min(len(des1), len(des2))
                            distances = [m.distance for m in good_matches]
                            avg_distance = float(np.mean(distances))
                            distance_score = max(0.0, 1.0 - (avg_distance / 200.0))
                            sift_score = 0.6 * match_ratio + 0.4 * distance_score
                            best_score = max(best_score, sift_score)
                            
                except Exception:
                    pass
            
            return min(1.0, best_score)
            
        except Exception:
            return 0.0

    @staticmethod
    def _similarity_template_matching(template: np.ndarray, test_roi: np.ndarray) -> float:
        """개선된 템플릿 매칭 유사도 계산"""
        try:
            # 크기 정규화
            if template.shape[0] > test_roi.shape[0] or template.shape[1] > test_roi.shape[1]:
                scale = min(test_roi.shape[0] / template.shape[0], test_roi.shape[1] / template.shape[1])
                new_w = max(1, int(template.shape[1] * scale))
                new_h = max(1, int(template.shape[0] * scale))
                template = cv2.resize(template, (new_w, new_h))
            
            # 여러 매칭 방법 시도
            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
            scores = []
            
            for method in methods:
                res = cv2.matchTemplate(test_roi, template, method)
                if method == cv2.TM_SQDIFF_NORMED:
                    # SQDIFF는 낮을수록 좋음
                    score = 1.0 - float(cv2.minMaxLoc(res)[0])
                else:
                    score = float(cv2.minMaxLoc(res)[1])
                scores.append(max(0.0, score))
            
            # 평균 점수 반환
            return float(np.mean(scores))
            
        except Exception:
            return 0.0

    @staticmethod
    def _similarity_histogram(template: np.ndarray, test_roi: np.ndarray) -> float:
        """개선된 히스토그램 유사도 계산"""
        try:
            # 정규화
            template_norm = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX)
            test_roi_norm = cv2.normalize(test_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            # 여러 히스토그램 비교 방법 사용
            methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
            scores = []
            
            for method in methods:
                hist1 = cv2.calcHist([template_norm], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([test_roi_norm], [0], None, [256], [0, 256])
                
                # 히스토그램 정규화
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                
                score = cv2.compareHist(hist1, hist2, method)
                
                # 각 방법에 따라 점수 변환
                if method == cv2.HISTCMP_CORREL:
                    score = max(0.0, score)  # 0~1 범위
                elif method == cv2.HISTCMP_CHISQR:
                    score = max(0.0, 1.0 - score / 100.0)  # 낮을수록 좋음
                elif method == cv2.HISTCMP_INTERSECT:
                    score = max(0.0, score)  # 0~1 범위
                elif method == cv2.HISTCMP_BHATTACHARYYA:
                    score = max(0.0, 1.0 - score)  # 낮을수록 좋음
                
                scores.append(score)
            
            # 가중 평균 (CORREL에 더 높은 가중치)
            weights = [0.4, 0.2, 0.2, 0.2]
            weighted_score = sum(w * s for w, s in zip(weights, scores))
            return float(min(1.0, weighted_score))  # 1.0을 넘지 않도록 제한
            
        except Exception:
            return 0.0

    @staticmethod
    def _similarity_ssim(template: np.ndarray, test_roi: np.ndarray) -> float:
        """구조적 유사도 지수 (SSIM) 계산"""
        try:
            from skimage.metrics import structural_similarity as ssim
            # 크기 맞추기
            if template.shape != test_roi.shape:
                test_roi = cv2.resize(test_roi, (template.shape[1], template.shape[0]))
            return float(ssim(template, test_roi))
        except ImportError:
            # skimage가 없으면 간단한 구조적 유사도 계산
            try:
                if template.shape != test_roi.shape:
                    test_roi = cv2.resize(test_roi, (template.shape[1], template.shape[0]))
                
                # 간단한 구조적 유사도 (평균, 분산, 공분산 기반)
                mu1 = np.mean(template)
                mu2 = np.mean(test_roi)
                sigma1 = np.var(template)
                sigma2 = np.var(test_roi)
                sigma12 = np.mean((template - mu1) * (test_roi - mu2))
                
                c1 = 0.01 ** 2
                c2 = 0.03 ** 2
                
                ssim_score = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                            ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
                return float(max(0.0, ssim_score))
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _similarity_edge(template: np.ndarray, test_roi: np.ndarray) -> float:
        """엣지 기반 유사도 계산"""
        try:
            # 크기 맞추기
            if template.shape != test_roi.shape:
                test_roi = cv2.resize(test_roi, (template.shape[1], template.shape[0]))
            
            # Canny 엣지 검출
            edges1 = cv2.Canny(template, 50, 150)
            edges2 = cv2.Canny(test_roi, 50, 150)
            
            # 엣지 픽셀 수 계산
            edge_count1 = np.sum(edges1 > 0)
            edge_count2 = np.sum(edges2 > 0)
            
            if edge_count1 == 0 and edge_count2 == 0:
                return 1.0
            if edge_count1 == 0 or edge_count2 == 0:
                return 0.0
            
            # 엣지 비율 유사도
            ratio_sim = 1.0 - abs(edge_count1 - edge_count2) / max(edge_count1, edge_count2)
            
            # 엣지 위치 유사도 (XOR 연산)
            xor_result = cv2.bitwise_xor(edges1, edges2)
            xor_count = np.sum(xor_result > 0)
            total_edges = max(edge_count1, edge_count2)
            position_sim = 1.0 - (xor_count / total_edges) if total_edges > 0 else 0.0
            
            # 가중 평균
            return float(0.6 * ratio_sim + 0.4 * position_sim)
            
        except Exception:
            return 0.0

    @staticmethod
    def _preprocess_image(img: np.ndarray) -> np.ndarray:
        """개선된 이미지 전처리"""
        try:
            # 1. 노이즈 제거 (가우시안 블러)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            
            # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # 3. 대비 향상 (적응적)
            mean_val = np.mean(enhanced)
            if mean_val < 100:  # 어두운 이미지
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
            elif mean_val > 180:  # 밝은 이미지
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            else:  # 일반적인 이미지
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            
            # 4. 샤프닝 적용 (엣지 강화)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            # 5. 히스토그램 스트레칭
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            return enhanced
        except Exception:
            return img

    @staticmethod
    def _similarity_robust(template: np.ndarray, test_roi: np.ndarray) -> float:
        """강건한 유사도 계산 (여러 방법의 조합)"""
        try:
            # 크기 맞추기
            if template.shape != test_roi.shape:
                test_roi = cv2.resize(test_roi, (template.shape[1], template.shape[0]))
            
            # 1. 템플릿 매칭 (여러 방법)
            tm_scores = []
            for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]:
                res = cv2.matchTemplate(test_roi, template, method)
                if method == cv2.TM_SQDIFF_NORMED:
                    score = 1.0 - float(cv2.minMaxLoc(res)[0])
                else:
                    score = float(cv2.minMaxLoc(res)[1])
                tm_scores.append(max(0.0, score))
            
            # 2. 히스토그램 유사도 (여러 방법)
            hist_scores = []
            for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]:
                hist1 = cv2.calcHist([template], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([test_roi], [0], None, [256], [0, 256])
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                score = cv2.compareHist(hist1, hist2, method)
                if method == cv2.HISTCMP_BHATTACHARYYA:
                    score = 1.0 - score
                hist_scores.append(max(0.0, score))
            
            # 3. 구조적 유사도 (SSIM)
            ssim_score = MissingPartInspector._similarity_ssim(template, test_roi)
            
            # 4. 엣지 유사도
            edge_score = MissingPartInspector._similarity_edge(template, test_roi)
            
            # 5. 픽셀 강도 유사도
            intensity_diff = np.abs(template.astype(float) - test_roi.astype(float))
            intensity_sim = 1.0 - (np.mean(intensity_diff) / 255.0)
            intensity_sim = max(0.0, intensity_sim)
            
            # 6. 텍스처 유사도 (LBP 기반)
            try:
                from skimage.feature import local_binary_pattern
                lbp1 = local_binary_pattern(template, 8, 1, method='uniform')
                lbp2 = local_binary_pattern(test_roi, 8, 1, method='uniform')
                hist1 = cv2.calcHist([lbp1.astype(np.uint8)], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([lbp2.astype(np.uint8)], [0], None, [256], [0, 256])
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                texture_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                texture_sim = max(0.0, texture_sim)
            except ImportError:
                texture_sim = 0.0
            
            # 가중 평균 (안정적인 방법들에 높은 가중치)
            weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]  # tm, hist, ssim, edge, intensity, texture, orb
            scores = [
                np.mean(tm_scores),
                np.mean(hist_scores),
                ssim_score,
                edge_score,
                intensity_sim,
                texture_sim,
                MissingPartInspector._similarity_orb(template, test_roi)  # ORB도 포함
            ]
            
            # 가중 평균 계산
            weighted_score = sum(w * s for w, s in zip(weights, scores))
            
            # 추가 보정: 높은 점수들에 보너스
            high_scores = [s for s in scores if s > 0.8]
            if len(high_scores) >= 3:  # 3개 이상의 방법이 높은 점수
                weighted_score = min(1.0, weighted_score * 1.1)
            
            # 낮은 점수들에 페널티 (하지만 너무 가혹하지 않게)
            low_scores = [s for s in scores if s < 0.3]
            if len(low_scores) >= 4:  # 4개 이상의 방법이 낮은 점수
                weighted_score = weighted_score * 0.9
            
            return float(min(1.0, weighted_score))
            
        except Exception:
            return 0.0

    @staticmethod
    def _adaptive_threshold(scores: dict, roi_id: int) -> float:
        """ROI별 적응적 임계값 계산"""
        try:
            # 기본 임계값 (더 관대하게)
            base_threshold = 0.65
            
            # 각 점수별 가중치
            weights = {
                'template': 0.35,
                'hist': 0.25,
                'ssim': 0.2,
                'edge': 0.15,
                'orb': 0.05
            }
            
            # 점수 기반 임계값 조정
            weighted_score = sum(weights.get(key, 0) * value for key, value in scores.items())
            
            # 점수 기반 동적 조정
            if weighted_score > 0.85:
                threshold = base_threshold + 0.1
            elif weighted_score > 0.7:
                threshold = base_threshold + 0.05
            elif weighted_score < 0.3:
                threshold = base_threshold - 0.2
            elif weighted_score < 0.5:
                threshold = base_threshold - 0.1
            else:
                threshold = base_threshold
            
            # ROI별 특별 조정 (더 세밀하게)
            if roi_id in [1, 3, 8, 9]:  # 작은 ROI들 - 더 관대하게
                threshold -= 0.08
            elif roi_id in [6]:  # 큰 ROI - 조금 더 엄격하게
                threshold += 0.03
            elif roi_id in [2, 4, 5, 7, 10, 11]:  # 중간 ROI들
                threshold -= 0.02
            
            # 안전 범위 내에서 제한
            return max(0.45, min(0.85, threshold))
            
        except Exception:
            return 0.65

    def inspect(self, test_image_path: str, method: str = "combined", display_scale: float = 1.2, save_image_path: str | None = None, show_window: bool = True):
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"검사 이미지 파일을 찾을 수 없습니다: {test_image_path}")
        if not self.templates:
            raise ValueError("템플릿이 로드되지 않았습니다. load_templates()를 먼저 호출하세요.")

        img = cv2.imread(test_image_path)
        if img is None:
            raise ValueError(f"검사 이미지를 읽을 수 없습니다: {test_image_path}")

        if self.normalize_size:
            norm_img, _, _ = self._normalize_image(img, self.target_size)
            gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
            vis = norm_img.copy()
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vis = img.copy()
        
        # 이미지 전처리 적용
        gray = self._preprocess_image(gray)

        results: Dict = {
            "image_path": test_image_path,
            "method": method,
            "threshold": self.threshold,
            "roi_results": [],
            "overall_result": "OK",
        }

        ng_count = 0
        placed_labels: list[tuple[int, int, int, int]] = []  # (x1,y1,x2,y2)
        for i, (x, y, w, h) in enumerate(self.roi_coords):
            roi = gray[y:y + h, x:x + w]
            template = self.templates[i + 1]
            
            # 템플릿도 전처리 적용
            template_processed = self._preprocess_image(template)

            # 모든 유사도 계산
            sim_orb = self._similarity_orb(template_processed, roi)
            sim_tm = self._similarity_template_matching(template_processed, roi)
            sim_hist = self._similarity_histogram(template_processed, roi)
            sim_ssim = self._similarity_ssim(template_processed, roi)
            sim_edge = self._similarity_edge(template_processed, roi)

            if method == "orb":
                sim = sim_orb
            elif method == "template":
                sim = sim_tm
            elif method == "histogram":
                sim = sim_hist
            elif method == "ssim":
                sim = sim_ssim
            elif method == "edge":
                sim = sim_edge
            elif method == "combined":
                # 개선된 가중 평균 (더 안정적인 방법들에 높은 가중치)
                weights = [0.2, 0.3, 0.25, 0.15, 0.1]  # orb, template, hist, ssim, edge
                scores = [sim_orb, sim_tm, sim_hist, sim_ssim, sim_edge]
                sim = sum(w * s for w, s in zip(weights, scores))
            elif method == "enhanced":
                # 가장 강건한 방법들만 사용
                weights = [0.3, 0.3, 0.2, 0.2]  # template, hist, ssim, edge
                scores = [sim_tm, sim_hist, sim_ssim, sim_edge]
                sim = sum(w * s for w, s in zip(weights, scores))
            elif method == "robust":
                # 강건한 방법 사용
                sim = self._similarity_robust(template_processed, roi)
            else:
                raise ValueError(f"지원하지 않는 검사 방법: {method}")

            # 적응적 임계값 적용
            if method in ["robust", "enhanced"]:
                adaptive_threshold = self._adaptive_threshold({
                    "template": sim_tm,
                    "hist": sim_hist,
                    "ssim": sim_ssim,
                    "edge": sim_edge,
                    "orb": sim_orb
                }, i + 1)
                is_ok = sim >= adaptive_threshold
            else:
                is_ok = sim >= self.threshold
            if not is_ok:
                ng_count += 1

            color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # 겹침/가독성 개선: ROI 크기에 따른 가변 폰트 크기와 배경 박스 적용 + 충돌 회피 배치
            font = cv2.FONT_HERSHEY_SIMPLEX
            # ROI 크기에 비례한 폰트 배율 (0.35 ~ 0.7)
            font_scale = max(0.35, min(0.7, min(w, h) / 220.0))
            thickness = 1 if font_scale <= 0.55 else 2
            label_text = f"R{i + 1}:{'OK' if is_ok else 'NG'} {sim:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            def clamp(p, lo, hi):
                return max(lo, min(hi, p))

            def intersects(r1, r2):
                (x1,y1,x2,y2),(a1,b1,a2,b2)=r1,r2
                return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)

            def first_non_overlapping_position() -> tuple[int,int]:
                W, H = vis.shape[1], vis.shape[0]
                margin = 4
                candidates = []
                # 1) inside top-left
                candidates.append((x + margin, y + margin + text_h))
                # 2) above box
                candidates.append((x, y - margin))  # we'll subtract text_h later
                # 3) below box
                candidates.append((x, y + h + margin + text_h))
                # 4) right side
                candidates.append((x + w + margin, y + text_h))
                # 5) left side
                candidates.append((x - text_w - margin, y + text_h))
                # 6) inside bottom-left
                candidates.append((x + margin, y + h - margin))

                for idx, (px, py) in enumerate(candidates):
                    # adjust for variants that used raw y
                    if idx == 1:  # above box: anchor baseline at y - margin
                        py = y - margin
                    # compute top-left of bg box
                    tx = clamp(px, 0, W - text_w - 4)
                    ty = clamp(py, text_h + 4, H - 4)
                    tl = (tx - 2, ty - text_h - 2)
                    br = (tx + text_w + 2, ty + baseline + 2)
                    rect = (tl[0], tl[1], br[0], br[1])
                    if all(not intersects(rect, r) for r in placed_labels):
                        placed_labels.append(rect)
                        return tx, ty
                # fallback: place at clamped top-left
                tx = clamp(x + 2, 0, W - text_w - 4)
                ty = clamp(y + 2 + text_h, text_h + 4, H - 4)
                rect = (tx - 2, ty - text_h - 2, tx + text_w + 2, ty + baseline + 2)
                placed_labels.append(rect)
                return tx, ty

            tx, ty = first_non_overlapping_position()

            # 배경 사각형 (검정)
            bg_tl = (tx - 2, ty - text_h - 2)
            bg_br = (tx + text_w + 2, ty + baseline + 2)
            cv2.rectangle(vis, bg_tl, bg_br, (0, 0, 0), thickness=-1)
            # 텍스트는 흰색으로, 테두리 색상은 박스 색 유지
            cv2.putText(vis, label_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            results["roi_results"].append({
                "roi_id": i + 1,
                "coordinates": (x, y, w, h),
                "similarity": float(sim),
                "scores": {
                    "orb": float(sim_orb), 
                    "template": float(sim_tm), 
                    "hist": float(sim_hist),
                    "ssim": float(sim_ssim),
                    "edge": float(sim_edge)
                },
                "is_ok": is_ok,
                "result": "OK" if is_ok else "NG",
            })

        if ng_count > 0:
            results["overall_result"] = "NG"

        disp = cv2.resize(vis, (0, 0), fx=display_scale, fy=display_scale)
        if save_image_path:
            try:
                os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
                cv2.imwrite(save_image_path, vis)
            except Exception:
                pass
        if show_window:
            cv2.imshow(f"검사 결과 - {method}", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return results, vis


