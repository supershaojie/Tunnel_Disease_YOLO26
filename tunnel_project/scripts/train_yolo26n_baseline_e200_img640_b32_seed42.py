from pathlib import Path

from ultralytics import YOLO


ROOT = Path("/root/autodl-tmp/projects/Tunnel_Disease_YOLO26")
MODEL = ROOT / "yolo26n.pt"
CONFIG = ROOT / "tunnel_project/configs/yolo26n_baseline_e200_img640_b32_seed42.yaml"


def main():
    assert MODEL.is_file(), f"缺少模型权重：{MODEL}"
    assert CONFIG.is_file(), f"缺少训练配置：{CONFIG}"

    model = YOLO(str(MODEL))

    # 空列表替换并关闭Ultralytics默认Albumentations。
    model.train(
        cfg=str(CONFIG),
        augmentations=[],
    )


if __name__ == "__main__":
    main()
