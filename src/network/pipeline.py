import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from src.network.atr import ATR
from src.network.lip import LIP

class HumanParsingPipeline:
    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        atr_repo_id: Optional[str] = None,
        atr_filename: Optional[str] = None,
        lip_repo_id: Optional[str] = None,
        lip_filename: Optional[str] = None,
    ) -> None:
        # 로거 설정
        self.logger = logging.getLogger("HumanParsingPipeline")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        # 핸들러가 없으면 추가
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("Initializing HumanParsingPipeline")
        
        # ATR & LIP 모델 초기화
        self.atr = ATR(
            device=device,
            verbose=verbose,
            repo_id=atr_repo_id,
            filename=atr_filename,
        )
        self.lip = LIP(
            device=device,
            verbose=verbose,
            repo_id=lip_repo_id,
            filename=lip_filename,
        )
        self.logger.info("Models initialized successfully")
        
    def __call__(self, image: np.ndarray):
        self.logger.info("Running HumanParsingPipeline")
        parsing_result = self.atr(image)
        parsing_result_lip = self.lip(image)
        
        # add neck parsing result
        neck_mask = np.logical_and(
            np.logical_not((parsing_result_lip == 13).astype(np.float32)),
            (parsing_result == 11).astype(np.float32),
        )
        parsing_result = np.where(neck_mask, 18, parsing_result)
        
        # 리팩토링된 부분: 팔레트 적용 및 시각화
        colormap = self.get_colormap(19)
        output_img = colormap[parsing_result.astype(np.uint8)]
        
        face_mask = (parsing_result == 11).astype(np.float32)
        return output_img, face_mask
        
    def get_colormap(self, num_cls: int):
        """
        Returns the color map for visualizing the segmentation mask using cv2 and numpy.
        
        Args:
            num_cls: Number of classes
            
        Returns:
            The color map as a numpy array with shape (num_cls, 3) containing BGR values
        """
        colormap = np.zeros((num_cls, 3), dtype=np.uint8)
        
        for j in range(num_cls):
            color = np.zeros(3, dtype=np.uint8)
            lab = j
            i = 0
            while lab and i < 8:  # 최대 8비트까지 처리
                color[0] |= (((lab >> 0) & 1) << (7 - i))
                color[1] |= (((lab >> 1) & 1) << (7 - i))
                color[2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
            
            # OpenCV에서는 BGR 순서로 저장
            colormap[j] = color[::-1]  # RGB to BGR
            
        return colormap