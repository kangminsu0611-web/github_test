#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
부품 누락 검사 스크립트 (OpenCV)

기능 개요
- 정상 이미지에서 ROI(관심영역)들을 선택하여 템플릿으로 저장
- 검사 이미지에서 동일 위치의 패턴 유사도를 계산하여 OK/NG 판정
- 검사 방법: ORB, 템플릿 매칭, 히스토그램, 또는 결합(combined)

사용 예시 (PowerShell)
1) ROI 설정:
   .\venv\Scripts\python.exe .\opencv.py --mode setup --normal 정상.jpg \
       --templates_dir templates --target-size 640 640 --normalize

2) 검사 실행:
   .\venv\Scripts\python.exe .\opencv.py --mode inspect --test 검사.jpg \
       --templates_dir templates --method combined --threshold 0.8
"""

import os
import json
import cv2
import argparse
import numpy as np
from typing import List, Tuple, Dict
from inspector import MissingPartInspector

DEFAULT_NORMAL_IMAGE = os.path.join("datasets", "nomal.jpg")
DEFAULT_TEST_IMAGE = os.path.join("datasets", "test.jpg")
DEFAULT_DISPLAY_SCALE = 1.5  # 코드에서 직접 조절할 표시 배율(ROI/결과 창 공통)


# MissingPartInspector 클래스는 inspector.py에서 import하여 사용


def main() -> None:
    parser = argparse.ArgumentParser(description="부품 누락 검사 (OpenCV)")
    parser.add_argument("--mode", choices=["setup", "inspect"], required=True, help="실행 모드")
    parser.add_argument("--normal", type=str, help="정상 이미지 경로 (setup)")
    parser.add_argument("--test", type=str, help="검사 이미지 경로 (inspect)")
    parser.add_argument("--method", choices=["orb", "template", "histogram", "ssim", "edge", "combined", "enhanced", "robust"], default="robust", help="검사 방법")
    parser.add_argument("--threshold", type=float, default=0.75, help="유사도 임계값 (0~1)")
    parser.add_argument("--templates_dir", type=str, default="templates", help="템플릿 저장 디렉터리")
    parser.add_argument("--normalize", action="store_true", default=True, help="이미지 크기 정규화 활성화")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="이미지 크기 정규화 비활성화")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="정규화 목표 크기")
    parser.add_argument("--display-scale", type=float, default=None, help="표시 배율 (0.1~2.0, ROI/결과 창)")

    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size else None
    inspector = MissingPartInspector(threshold=args.threshold, normalize_size=args.normalize, target_size=target_size)

    # 터미널 인자가 없을 때는 코드 상수(DEFAULT_DISPLAY_SCALE)를 사용
    effective_scale = DEFAULT_DISPLAY_SCALE if args.display_scale is None else args.display_scale
    effective_scale = max(0.1, min(2.0, effective_scale))

    if args.mode == "setup":
        if not args.normal:
            if os.path.exists(DEFAULT_NORMAL_IMAGE):
                args.normal = DEFAULT_NORMAL_IMAGE
                print(f"기본 정상 이미지 사용: {args.normal}")
            else:
                args.normal = input("정상 이미지 경로를 입력하세요: ").strip().strip('"')
                if not args.normal:
                    print("정상 이미지 경로가 필요합니다.")
                    return
        rois = inspector.select_rois(args.normal, display_scale=effective_scale)
        if not rois:
            print("ROI가 선택되지 않았습니다.")
            return
        inspector.save_templates(args.normal, rois, args.templates_dir)
        print("ROI 템플릿 저장 완료.")
    else:
        if not args.test:
            if os.path.exists(DEFAULT_TEST_IMAGE):
                args.test = DEFAULT_TEST_IMAGE
                print(f"기본 검사 이미지 사용: {args.test}")
            else:
                args.test = input("검사 이미지 경로를 입력하세요: ").strip().strip('"')
                if not args.test:
                    print("검사 이미지 경로가 필요합니다.")
                    return
        inspector.load_templates(args.templates_dir)
        results, vis = inspector.inspect(args.test, args.method, display_scale=effective_scale, save_image_path=None, show_window=False)
        
        # 결과 표시
        if effective_scale <= 0:
            effective_scale = 1.0
        disp = cv2.resize(vis, (0, 0), fx=effective_scale, fy=effective_scale)
        try:
            cv2.imwrite("inspection_result.jpg", vis)
        except Exception:
            pass
        cv2.imshow(f"검사 결과 - {args.method}", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        out = os.path.join(os.getcwd(), "inspection_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"검사 완료 → {out}")


if __name__ == "__main__":
    main()


