#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI 설정 전용 CLI

사용 예시 (PowerShell)
1) ROI 설정:
명령어 : .\venv\Scripts\python.exe .\roi_setup.py
"""

import argparse
import os
from inspector import MissingPartInspector

DEFAULT_NORMAL_IMAGE = os.path.join("datasets", "nomal.jpg")
DEFAULT_DISPLAY_SCALE = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI 설정 전용 CLI")
    parser.add_argument("--normal", type=str, help="정상 이미지 경로")
    parser.add_argument("--templates_dir", type=str, default="templates", help="템플릿 저장 디렉터리")
    parser.add_argument("--display-scale", type=float, default=DEFAULT_DISPLAY_SCALE, help="표시 배율 (0.1~2.0)")
    parser.add_argument("--threshold", type=float, default=0.8, help="유사도 임계값(저장에는 영향 없음)")
    parser.add_argument("--normalize", action="store_true", default=True, help="이미지 크기 정규화 활성화")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="이미지 크기 정규화 비활성화")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="정규화 목표 크기")

    args = parser.parse_args()

    if not args.normal:
        args.normal = DEFAULT_NORMAL_IMAGE if os.path.exists(DEFAULT_NORMAL_IMAGE) else input("정상 이미지 경로: ").strip().strip('"')
    target_size = tuple(args.target_size) if args.target_size else None

    inspector = MissingPartInspector(threshold=args.threshold, normalize_size=args.normalize, target_size=target_size)
    rois = inspector.select_rois(args.normal, display_scale=max(0.1, min(2.0, args.display_scale)))
    if not rois:
        print("ROI가 선택되지 않았습니다.")
        return
    inspector.save_templates(args.normal, rois, args.templates_dir)
    print("ROI 템플릿 저장 완료.")


if __name__ == "__main__":
    main()


