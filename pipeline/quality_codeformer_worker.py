"""Run inside the official CodeFormer checkout's Python environment."""
from pathlib import Path
import sys


def main(root, folder, device, fidelity):
    sys.path.insert(0, root)
    import cv2
    import numpy as np
    import torch
    from basicsr.archs.codeformer_arch import CodeFormer
    from basicsr.utils import img2tensor, tensor2img
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import is_gray
    from torchvision.transforms.functional import normalize

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CodeFormer 环境没有可用的 CUDA 设备")
    folder = Path(folder)
    image = cv2.imdecode(np.frombuffer((folder / "input.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
                               save_ext="png", use_parse=True, device=torch.device(device))
    # Keep the original canvas for pasting, since read_image enlarges small backgrounds.
    helper.input_img = image
    helper.is_gray = is_gray(image, threshold=10)
    h, w = image.shape[:2]
    detection_size = min(640, max(1, round(1600 * min(h, w) / max(h, w))))
    with torch.inference_mode():
        helper.get_face_landmarks_5(only_center_face=False, resize=detection_size, eye_dist_threshold=12)
    helper.align_warp_face()
    count = len(helper.cropped_faces)
    if count:
        model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                           connect_list=["32", "64", "128", "256"])
        weights = torch.load("weights/CodeFormer/codeformer.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(weights["params_ema"])
        model = model.eval().to(device)
        with torch.inference_mode():
            for face in helper.cropped_faces:
                tensor = img2tensor(face / 255., bgr2rgb=True, float32=True)
                normalize(tensor, [.5] * 3, [.5] * 3, inplace=True)
                result = model(tensor.unsqueeze(0).to(device), w=float(fidelity), adain=True)[0]
                restored = tensor2img(result, rgb2bgr=True, min_max=(-1, 1))
                # Keep source structure and texture with a conservative aligned-face blend.
                helper.add_restored_face(cv2.addWeighted(restored, .65, face, .35, 0), face)
            helper.get_inverse_affine(None)
            result = helper.paste_faces_to_input_image()
        ok, data = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("CodeFormer 图片编码失败")
        (folder / "output.png").write_bytes(data.tobytes())
    (folder / "faces.txt").write_text(str(count), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
