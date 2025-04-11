import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List, Union, Sequence
from src.network.base import ONNXWrapper


class LIP(ONNXWrapper):
    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the LIP model wrapper.

        Args:
            device (str): Target device ('cpu', 'cuda', or 'cuda:device_id'). Defaults to 'cuda'.
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING. Defaults to False.
            repo_id (str, optional): Hugging Face Hub repository ID. Required if using Hugging Face.
            filename (str, optional): Filename of the model in the repository. Required if using Hugging Face.
            model_path (str, optional): Direct path to the model file. Used if repo_id and filename are not provided.
        """
        super().__init__(device, verbose, repo_id, filename, model_path)

    def infer(self, image: np.ndarray) -> np.ndarray:
        self.logger.info("Running inference on LIP model")
        # Preprocess the image
        processed_img, meta = self.preprocess(image)

        # ONNX 세션에 입력 준비
        input_name = self.input_names[0]
        ort_input = {input_name: processed_img}

        # Inference
        ort_outputs = self.session.run(None, ort_input)

        # 출력 후처리
        output = self.postprocess(ort_outputs, meta)
        return output

    def postprocess(
        self, outputs: List[np.ndarray], meta: Dict[str, Any]
    ) -> np.ndarray:
        """
        모델 출력을 후처리하여 최종 파싱 결과를 생성합니다.

        Args:
            outputs: 모델의 출력 텐서 리스트
            meta: 전처리 단계에서 생성된 메타데이터

        Returns:
            np.ndarray: 각 픽셀이 인체 부위 클래스 ID를 나타내는 분할 마스크
        """
        self.logger.info("Postprocessing ATR model output")
        output = outputs[1][0]
        output = np.transpose(output, (1, 2, 0))

        # 이중선형 보간법(bilinear)을 사용하여 업샘플링
        output = cv2.resize(
            output,
            (self.model_input_shape[0], self.model_input_shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )

        # 로짓 변환 적용
        parsing_result_lip = self._transform_logits(output, meta)
        parsing_result_lip = np.argmax(parsing_result_lip, axis=2)
        return parsing_result_lip

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        인퍼런스를 위한 이미지 전처리 함수.

        수행 과정:
          - 원본 이미지 전체 영역을 대상으로 center와 scale을 계산
          - ONNX 모델의 입력 크기에 맞게 affine 변환 행렬 계산
          - 이미지를 지정된 크기로 변환 및 정규화
          - 채널 순서 변경 및 배치 차원 추가

        Args:
            image: cv2로 읽은 원본 이미지 (BGR 순서)

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 전처리된 이미지와
                                               center, original height/width, scale, rotation을 담은 meta 딕셔너리
        """
        self.logger.info("Preprocessing image for ATR model")
        # ONNX 모델이 기대하는 입력 크기 사용 (width, height 순)
        dst_w, dst_h = self.model_input_shape
        aspect_ratio = dst_w / dst_h

        # 원본 이미지 크기 (h: height, w: width)
        h, w = image.shape[:2]

        # 전체 이미지를 대상으로 bounding box 정의
        x, y = 0, 0
        box_w, box_h = w - 1, h - 1

        # 이미지 중심 계산
        center = np.array([x + box_w * 0.5, y + box_h * 0.5], dtype=np.float32)

        # 원본 이미지의 bounding box 크기를 모델의 aspect ratio에 맞게 조정
        if box_w > aspect_ratio * box_h:
            box_h = box_w / aspect_ratio
        elif box_w < aspect_ratio * box_h:
            box_w = box_h * aspect_ratio
        scale = np.array([box_w, box_h], dtype=np.float32)

        # 회전 값 (현재 0으로 고정)
        rotation = 0

        # affine 변환 행렬 계산:
        # - scale_x, scale_y: 각각 가로, 세로 방향의 스케일링 비율
        # - trans_x, trans_y: center를 모델 입력의 중앙으로 옮기기 위한 translation 값
        scale_x = dst_w / scale[0]
        scale_y = dst_h / scale[1]
        trans_x = dst_w / 2 - center[0] * scale_x
        trans_y = dst_h / 2 - center[1] * scale_y
        trans_mat = np.array(
            [[scale_x, 0, trans_x], [0, scale_y, trans_y]], dtype=np.float32
        )

        # cv2.warpAffine을 이용하여 affine 변환 적용
        processed_img = cv2.warpAffine(
            image,
            trans_mat,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # 이미지 정규화에 사용할 평균 및 표준편차 값
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 이미지 정규화 적용
        processed_img = (processed_img.astype(np.float32) - mean) / std
        # HWC -> CHW 변환 (채널 우선 순서로 변경)
        processed_img = np.transpose(processed_img, (2, 0, 1))
        # 배치 차원 추가
        processed_img = np.expand_dims(processed_img, axis=0)

        # meta 정보 구성: 전처리 과정에서 사용한 center, 원본 이미지 크기, scale, rotation 등을 저장
        meta = {
            "center": center,
            "height": h,
            "width": w,
            "scale": scale,
            "rotation": rotation,
        }

        return processed_img, meta

    def _transform_logits(self, logits: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        """
        로짓 결과를 원본 이미지 크기로 역변환합니다.

        Args:
            logits: 모델에서 출력된 로짓 결과
            meta: 전처리 단계에서 생성된 메타데이터

        Returns:
            np.ndarray: 원본 이미지 크기로 변환된 로짓 결과
        """
        c = meta["center"]
        s = meta["scale"]
        w = meta["width"]
        h = meta["height"]
        # 역 affine 변환 행렬 계산
        trans = self.get_affine_transform(c, s, 0, self.model_input_shape, inv=1)
        channel = logits.shape[2]
        target_logits = []
        # 각 채널별로 역변환 적용
        for i in range(channel):
            target_logit = cv2.warpAffine(
                logits[:, :, i],
                trans,
                (int(w), int(h)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0),
            )
            target_logits.append(target_logit)
        # 채널 방향으로 다시 스택
        target_logits = np.stack(target_logits, axis=2)
        return target_logits

    def get_affine_transform(
        self,
        center: np.ndarray,
        scale: Union[np.ndarray, List[float], Tuple[float, float]],
        rot: float,
        output_size: Tuple[int, int],
        shift: np.ndarray = np.array([0, 0], dtype=np.float32),
        inv: int = 0,
    ) -> np.ndarray:
        """
        Affine 변환 행렬을 계산합니다.

        Args:
            center: 변환 중심점 [x, y]
            scale: 크기 정보 [width, height]
            rot: 회전 각도 (도 단위)
            output_size: 출력 이미지 크기 (height, width)
            shift: 추가 이동 벡터
            inv: 역변환 행렬 계산 여부 (1: 역변환, 0: 정변환)

        Returns:
            np.ndarray: 2x3 affine 변환 행렬
        """
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale

        src_w = scale_tmp[0]
        dst_w = output_size[1]
        dst_h = output_size[0]

        # 회전 라디안 변환
        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

        # 소스 및 대상 포인트 설정
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
        dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

        # 세 번째 점 계산 (세 점으로 affine 변환 행렬 정의)
        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        # 정변환 또는 역변환 행렬 계산
        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def affine_transform(self, pt: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        점에 affine 변환을 적용합니다.

        Args:
            pt: 변환할 점 [x, y]
            t: 2x3 affine 변환 행렬

        Returns:
            np.ndarray: 변환된 점 [x', y']
        """
        new_pt = np.array([pt[0], pt[1], 1.0]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        두 점으로부터 세 번째 점을 계산합니다.
        세 점은 직각 삼각형을 형성합니다.

        Args:
            a: 첫 번째 점 [x1, y1]
            b: 두 번째 점 [x2, y2]

        Returns:
            np.ndarray: 세 번째 점 [x3, y3]
        """
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(self, src_point: Sequence[float], rot_rad: float) -> List[float]:
        """
        주어진 점을 회전시킵니다.

        Args:
            src_point: 회전시킬 점 [x, y]
            rot_rad: 회전 각도 (라디안 단위)

        Returns:
            List[float]: 회전된 점 [x', y']
        """
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def hole_fill(self, img: np.ndarray) -> np.ndarray:
        """
        이미지의 구멍을 채웁니다.

        플러드 필(Flood Fill) 알고리즘을 사용하여 배경을 채운 후,
        원본과의 비트 연산으로 구멍을 채웁니다.

        Args:
            img: 처리할 이미지 (바이너리 형태)

        Returns:
            np.ndarray: 구멍이 채워진 이미지
        """
        img_copy = img.copy()
        # 플러드 필을 위한 마스크 생성 (패딩 추가)
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        # 이미지의 (0,0) 위치에서 시작하여 연결된 영역을 255로 채움 (배경 채우기)
        cv2.floodFill(img, mask, (0, 0), 255)
        # 이미지 반전
        img_inverse = cv2.bitwise_not(img)
        # 원본 이미지와 반전된 이미지의 OR 연산 (구멍 채우기)
        dst = cv2.bitwise_or(img_copy, img_inverse)
        return dst

    def refine_hole(
        self,
        parsing_result_filled: np.ndarray,
        parsing_result: np.ndarray,
        arm_mask: np.ndarray,
    ) -> np.ndarray:
        """
        의류와 팔 사이의 빈 공간을 정제합니다.

        특정 크기 이상의 구멍만 보존하여 노이즈를 제거합니다.

        Args:
            parsing_result_filled: 구멍이 채워진 파싱 결과
            parsing_result: 원본 파싱 결과
            arm_mask: 팔 부위 마스크

        Returns:
            np.ndarray: 정제된 구멍 마스크
        """
        # 채워진 상의 영역에서 원래 상의가 아니었던 부분 추출 (팔 부분 제외)
        filled_hole = (
            cv2.bitwise_and(
                np.where(parsing_result_filled == 4, 255, 0),
                np.where(parsing_result != 4, 255, 0),
            )
            - arm_mask * 255
        )
        # 윤곽선 찾기
        contours, hierarchy = cv2.findContours(
            filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
        )
        refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)

        # 특정 크기(2000픽셀) 이상의 구멍만 유지
        for i in range(len(contours)):
            a = cv2.contourArea(contours[i], True)
            if abs(a) > 2000:
                cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)

        # 팔 마스크와 결합
        return refine_hole_mask + arm_mask
