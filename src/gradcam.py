import cv2
import numpy as np
import torch


class GradCAM:
    """Minimal GradCAM helper for torchvision ResNet-like models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_cam_on_image(img_rgb, cam_map, alpha=0.4):
    """Blend a normalized CAM map with an RGB image."""
    cam_u8 = np.uint8(cam_map * 255.0)
    cam_u8 = cv2.resize(cam_u8, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap, alpha, 0)
    return blended
