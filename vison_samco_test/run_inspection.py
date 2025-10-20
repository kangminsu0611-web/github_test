#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검사 실행 전용 CLI

기본 사용법:
- 코드 상단의 DEFAULT_FOLDER 변수에 검사할 폴더 경로를 설정한 후 실행
- 기본값: datasets 폴더 내 모든 이미지 검사

사용 예시 (PowerShell)
1) 기본 실행 (코드에서 설정한 폴더 검사):
명령어 : .\\venv\\Scripts\\python.exe .\\run_inspection.py --method combined --threshold 0.5

2) 특정 폴더 지정:
명령어 : .\\venv\\Scripts\\python.exe .\\run_inspection.py --folder my_images --method combined --threshold 0.5

3) 단일 이미지 검사:
명령어 : .\\venv\\Scripts\\python.exe .\\run_inspection.py --test datasets/test2.jpg --method combined --threshold 0.5
"""
import argparse
import os
import json
import glob
import time
from typing import List, Tuple
from inspector import MissingPartInspector

DEFAULT_TEST_IMAGE = os.path.join("datasets", "test2.jpg")
DEFAULT_DISPLAY_SCALE = 1.5

# =============================================================================
# 폴더 설정 - 여기서 검사할 폴더를 지정하세요
# =============================================================================
DEFAULT_FOLDER = "test_data"  # 검사할 이미지들이 있는 폴더 경로
# DEFAULT_FOLDER = "C:/Users/pc/Desktop/my_images"  # 절대 경로도 가능
# DEFAULT_FOLDER = "result/OK"  # 다른 폴더도 가능
# =============================================================================

# 지원하는 이미지 확장자
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


def find_image_files(folder_path: str) -> List[str]:
    """폴더 내의 모든 이미지 파일을 찾습니다."""
    image_files = []
    
    if not os.path.exists(folder_path):
        print(f"경고: 폴더 '{folder_path}'가 존재하지 않습니다.")
        return image_files
    
    for ext in SUPPORTED_EXTENSIONS:
        # 대소문자 구분 없이 검색
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # 중복 제거 및 정렬
    image_files = sorted(list(set(image_files)))
    return image_files


def get_next_count_path(result_root: str, target_dir: str, base: str) -> str:
    """다음 카운트 번호를 가진 파일 경로를 생성합니다.

    동작: result_root 아래(OK/NG/JSON 포함)의 모든 파일을 검사하여 동일한 base 이름으로 생성된
    최대 인덱스를 찾고, 그 다음 인덱스로 파일명을 만듭니다. 이렇게 하면 OK/NG 폴더 구분과
    관계없이 결과 이미지들이 항상 동일한 전역 인덱스를 공유하게 됩니다.
    """
    import re
    pattern = re.compile(re.escape(base) + r"_result_(\d+)\.(jpg|jpeg|png|json)$", re.IGNORECASE) # jpg
    max_idx = 0

    if os.path.exists(result_root):
        for root, _, files in os.walk(result_root):
            for name in files:
                m = pattern.match(name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        pass

    return os.path.join(target_dir, f"{base}_result_{max_idx + 1:04d}.jpg")


def process_single_image(inspector: MissingPartInspector, image_path: str, method: str, 
                        display_scale: float, result_root: str) -> Tuple[dict, str, str]:
    """단일 이미지를 검사하고 결과를 저장합니다."""
    ok_dir = os.path.join(result_root, "OK")
    ng_dir = os.path.join(result_root, "NG")
    json_dir = os.path.join(result_root, "JSON")
    
    # 검사 수행
    results, vis = inspector.inspect(
        image_path,
        method,
        display_scale=display_scale,
        save_image_path=None,
        show_window=False,
    )
    
    # 전체 판정: NG가 하나라도 있으면 NG
    final_label = "NG" if any(not r["is_ok"] for r in results["roi_results"]) else "OK"
    
    # 파일 이름 생성
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    target_dir = ng_dir if final_label == "NG" else ok_dir
    
    image_out_path = get_next_count_path(result_root, target_dir, base_name)
    json_out_path = os.path.join(json_dir, os.path.splitext(os.path.basename(image_out_path))[0] + ".json")
    
    # 시각화 결과 저장
    try:
        import cv2
        os.makedirs(os.path.dirname(image_out_path), exist_ok=True)
        cv2.imwrite(image_out_path, vis)
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
    
    # JSON 저장
    try:
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"JSON 저장 실패: {e}")
    
    return results, image_out_path, json_out_path, final_label


def process_batch_images(inspector: MissingPartInspector, image_files: List[str], method: str,
                        display_scale: float, result_root: str, show_progress: bool = True) -> dict:
    """여러 이미지를 배치로 처리합니다."""
    total_images = len(image_files)
    ok_count = 0
    ng_count = 0
    start_time = time.time()
    
    print(f"총 {total_images}개의 이미지를 검사합니다...")
    
    for i, image_path in enumerate(image_files, 1):
        if show_progress:
            print(f"\n[{i}/{total_images}] 처리 중: {os.path.basename(image_path)}")
        
        try:
            results, image_out_path, json_out_path, final_label = process_single_image(
                inspector, image_path, method, display_scale, result_root
            )
            
            if final_label == "OK":
                ok_count += 1
            else:
                ng_count += 1
            
            if show_progress:
                print(f"  → 결과: {final_label} | 저장: {os.path.basename(image_out_path)}")
                
        except Exception as e:
            print(f"  → 오류: {e}")
            ng_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 최종 통계
    stats = {
        "total_images": total_images,
        "ok_count": ok_count,
        "ng_count": ng_count,
        "processing_time": processing_time,
        "avg_time_per_image": processing_time / total_images if total_images > 0 else 0
    }
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="검사 실행 전용 CLI")
    
    # 입력 소스 선택 (상호 배타적)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--test", type=str, help="검사할 단일 이미지 경로")
    input_group.add_argument("--folder", type=str, help="검사할 이미지들이 있는 폴더 경로")
    
    # 검사 설정
    parser.add_argument("--templates_dir", type=str, default="templates", help="템플릿 저장 디렉터리")
    parser.add_argument("--method", choices=["orb", "template", "histogram", "ssim", "edge", "combined", "enhanced", "robust"], default="robust", help="검사 방법")
    parser.add_argument("--threshold", type=float, default=0.75, help="유사도 임계값")
    parser.add_argument("--display-scale", type=float, default=DEFAULT_DISPLAY_SCALE, help="표시 배율 (0.1~2.0)")
    
    # 출력 옵션
    parser.add_argument("--show", action="store_true", default=True, help="결과 창 표시 (단일 이미지 검사 시)")
    parser.add_argument("--no-show", action="store_false", dest="show", help="결과 창 표시 안 함")
    parser.add_argument("--no-progress", action="store_true", help="배치 처리 시 진행 상황 표시 안 함")
    
    # 이미지 처리 옵션
    parser.add_argument("--normalize", action="store_true", default=True, help="이미지 크기 정규화 활성화")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="이미지 크기 정규화 비활성화")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="정규화 목표 크기")

    args = parser.parse_args()

    # 입력 소스 결정
    if args.folder:
        # 명령행에서 폴더 지정된 경우
        folder_path = args.folder
        image_files = find_image_files(folder_path)
        if not image_files:
            print(f"오류: '{folder_path}' 폴더에서 이미지 파일을 찾을 수 없습니다.")
            print(f"지원하는 확장자: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        print(f"'{folder_path}' 폴더에서 {len(image_files)}개의 이미지를 찾았습니다.")
        single_image_mode = False
    elif args.test:
        # 단일 이미지 모드
        if not os.path.exists(args.test):
            print(f"오류: 이미지 파일 '{args.test}'이 존재하지 않습니다.")
            return
        image_files = [args.test]
        single_image_mode = True
    else:
        # 기본값: 코드에서 설정한 폴더 검사
        folder_path = DEFAULT_FOLDER
        image_files = find_image_files(folder_path)
        if not image_files:
            print(f"오류: 기본 폴더 '{folder_path}'에서 이미지 파일을 찾을 수 없습니다.")
            print(f"코드 상단의 DEFAULT_FOLDER 값을 확인하거나 --test 옵션으로 단일 이미지를 지정하세요.")
            print(f"지원하는 확장자: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        print(f"기본 폴더 '{folder_path}'에서 {len(image_files)}개의 이미지를 찾았습니다.")
        single_image_mode = False

    # 검사기 초기화
    target_size = tuple(args.target_size) if args.target_size else None
    inspector = MissingPartInspector(threshold=args.threshold, normalize_size=args.normalize, target_size=target_size)
    inspector.load_templates(args.templates_dir)
    
    # 결과 폴더 구성
    result_root = os.path.join(os.getcwd(), "result")
    ok_dir = os.path.join(result_root, "OK")
    ng_dir = os.path.join(result_root, "NG")
    json_dir = os.path.join(result_root, "JSON")
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(ng_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    display_scale = max(0.1, min(2.0, args.display_scale))

    if single_image_mode:
        # 단일 이미지 검사
        image_path = image_files[0]
        print(f"단일 이미지 검사: {os.path.basename(image_path)}")
        
        results, image_out_path, json_out_path, final_label = process_single_image(
            inspector, image_path, args.method, display_scale, result_root
        )
        
        print(f"검사 완료 → 이미지: {os.path.basename(image_out_path)} | JSON: {os.path.basename(json_out_path)} | 최종: {final_label}")

        # 결과 창 표시 옵션
        if args.show:
            try:
                import cv2
                _, vis = inspector.inspect(
                    image_path,
                    args.method,
                    display_scale=display_scale,
                    save_image_path=None,
                    show_window=False,
                )
                ds = max(0.1, min(2.0, args.display_scale))
                disp = cv2.resize(vis, (0, 0), fx=ds, fy=ds)
                cv2.imshow(f"검사 결과 - {args.method} ({final_label})", disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"결과 창 표시 실패: {e}")
    
    else:
        # 배치 이미지 검사
        stats = process_batch_images(
            inspector, image_files, args.method, display_scale, result_root, 
            show_progress=not args.no_progress
        )
        
        # 최종 통계 출력
        print("\n" + "="*50)
        print("배치 검사 완료!")
        print(f"총 이미지 수: {stats['total_images']}")
        print(f"OK: {stats['ok_count']} ({stats['ok_count']/stats['total_images']*100:.1f}%)")
        print(f"NG: {stats['ng_count']} ({stats['ng_count']/stats['total_images']*100:.1f}%)")
        print(f"총 처리 시간: {stats['processing_time']:.2f}초")
        print(f"평균 처리 시간: {stats['avg_time_per_image']:.2f}초/이미지")
        print(f"결과 저장 위치: {result_root}")
        print("="*50)


if __name__ == "__main__":
    main()


