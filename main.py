from src.network.pipeline import HumanParsingPipeline
from src.network.atr import ATR
from src.network.lip import LIP
from src.utils.loader import ImageLoader

# Constants
REPO_ID = "Longcat2957/humanparsing-onnx"


if __name__ == "__main__":
    il = ImageLoader(verbose=True)

    image = il.load("sample.png")

    # Initialize the pipeline
    hpp = HumanParsingPipeline(
        device="cuda:0",
        verbose=True,
        atr_repo_id=REPO_ID,
        lip_repo_id=REPO_ID,
        atr_filename="parsing_atr.onnx",
        lip_filename="parsing_lip.onnx",
    )

    somethings = hpp(image)
