# YOLO-NAS 기반 객체 탐지 프로젝트

이 프로젝트는 [YOLO-NAS](https://docs.ultralytics.com/ko/models/yolo-nas/#supported-tasks-and-modes) 모델을 기반으로, UAV-Human 데이터셋을 활용하여 사람 및 객체를 탐지하는 실험을 진행합니다.

## 프로젝트 개요

- 목적: 드론 시점에서 촬영된 영상에서 사람을 정밀하게 탐지
- 모델: YOLO-NAS (`yolo_nas_l.pt`)
- 데이터셋: UAV-Human

## 데이터셋

## 경로 설명

- `image_folder`는 **객체 탐지에 사용할 이미지 데이터셋 폴더** 경로입니다.  
  예:  
  ```python
  image_folder = "Dataset/M1003"
  ```

- `result_path`는 **탐지 결과 영상이 저장될 위치**를 지정합니다.  
  예:  
  ```python
  result_path = "Result/New_M1003.mp4"
  ```

> 이 두 경로는 상대경로 기준으로 설정되어 있으며, 실행 전 해당 폴더들이 존재하는지 확인해야 합니다.

### 원본 링크
- GitHub: https://github.com/SUTDCV/UAV-Human

### Google Drive 다운로드
- https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs
