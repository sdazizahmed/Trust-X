"""
TRUST-X core — model loading + inference functions.
Imported by both Streamlit pages and notebooks.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from scipy.optimize import minimize_scalar

# ── Constants ───────────────────────────────────────────────────────────────
LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax',
]

# Clinical severity weights (higher = more urgent)
SEVERITY_WEIGHTS = {
    'Pneumothorax'      : 5.0,
    'Pneumonia'         : 4.0,
    'Edema'             : 4.0,
    'Consolidation'     : 3.5,
    'Mass'              : 3.5,
    'Effusion'          : 3.0,
    'Cardiomegaly'      : 3.0,
    'Nodule'            : 2.5,
    'Atelectasis'       : 2.5,
    'Infiltration'      : 2.0,
    'Emphysema'         : 2.0,
    'Fibrosis'          : 1.5,
    'Pleural_Thickening': 1.5,
    'Hernia'            : 1.0,
}
SEVERITY_VEC = np.array([SEVERITY_WEIGHTS[l] for l in LABEL_COLS])


# ── Model ───────────────────────────────────────────────────────────────────
def build_densenet(num_classes, dropout=0.3):
    """DenseNet-121 with custom classifier head."""
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_trustx_model(checkpoint_path, device='cpu'):
    """Load TRUST-X checkpoint. Returns model, metadata dict, inference transform."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_densenet(
        num_classes=len(ckpt['label_cols']),
        dropout=ckpt.get('dropout', 0.3)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    norm = ckpt['normalisation']
    img_size = ckpt['img_size']
    infer_tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=norm['mean'], std=norm['std']),
    ])

    metadata = {
        'label_cols'    : ckpt['label_cols'],
        'img_size'      : img_size,
        'val_auc'       : ckpt.get('val_mean_auc'),
        'test_auc'      : ckpt.get('test_mean_auc'),
        'architecture'  : ckpt.get('architecture', 'densenet121'),
        'device'        : device,
    }
    return model, metadata, infer_tfm


# ── Temperature calibration ─────────────────────────────────────────────────
def fit_temperatures(val_logits, val_labels, label_cols):
    """Fit per-label temperature scalars. Returns np.array of shape [n_labels]."""
    def _find_T(logits, labels):
        def nll(T):
            scaled = torch.sigmoid(torch.tensor(logits / T, dtype=torch.float32))
            return nn.BCELoss()(scaled,
                                torch.tensor(labels, dtype=torch.float32)).item()
        return minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded').x

    return np.array([_find_T(val_logits[:, i], val_labels[:, i])
                     for i in range(len(label_cols))])


# ── Inference ───────────────────────────────────────────────────────────────
def predict(model, image, infer_tfm, device, temperatures=None, calibrate=True):
    """
    Single-image prediction. Returns np.array of probabilities [n_labels].
    `image` can be a PIL Image or a path.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    x = infer_tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).cpu().squeeze().numpy()

    if calibrate and temperatures is not None:
        logits = logits / temperatures
    return torch.sigmoid(torch.tensor(logits)).numpy()


# ── MC Dropout uncertainty ──────────────────────────────────────────────────
def mc_dropout_predict(model, image, infer_tfm, device,
                       temperatures=None, calibrate=True, n_passes=20):
    """
    Monte Carlo Dropout. Returns (mean_probs, std_probs, confidence_tiers).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    x = infer_tfm(image).unsqueeze(0).to(device)

    # Enable dropout at inference
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x).cpu().squeeze().numpy()
            if calibrate and temperatures is not None:
                logits = logits / temperatures
            all_probs.append(torch.sigmoid(torch.tensor(logits)).numpy())
    model.eval()

    all_probs = np.array(all_probs)
    mean_prob = all_probs.mean(axis=0)
    std_prob  = all_probs.std(axis=0)

    def _tier(s):
        if s < 0.05: return 'HIGH'
        if s < 0.10: return 'MEDIUM'
        return 'LOW'
    confidence = [_tier(s) for s in std_prob]

    return mean_prob, std_prob, confidence


# ── GradCAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """GradCAM for DenseNet-121 — targets last conv layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target = logits[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def build_gradcam(model):
    """Returns a configured GradCAM targeting denseblock4.denselayer16.conv2."""
    target_layer = model.features.denseblock4.denselayer16.conv2
    return GradCAM(model, target_layer)


def gradcam_heatmap(gradcam, image, infer_tfm, device, class_idx, img_size):
    """Generate heatmap numpy array [img_size, img_size] for given class."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    x = infer_tfm(image).unsqueeze(0).to(device)
    x.requires_grad_(True)
    return gradcam.generate(x, int(class_idx))


def overlay_heatmap(pil_image, cam, img_size, alpha=0.45):
    """Overlay CAM on image. Returns np.array RGB."""
    import cv2
    img_np = np.array(pil_image.resize((img_size, img_size)))
    cam_resized = cv2.resize(cam, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img_np)


# ── Triage ──────────────────────────────────────────────────────────────────
def triage_score(probs):
    """Compute severity-weighted triage score from probability vector."""
    return float((probs * SEVERITY_VEC).sum())


def priority_tier(score):
    """Return (label, colour_hex) for a given triage score."""
    if score > 8.0: return ('URGENT',  '#c0392b')
    if score > 4.0: return ('HIGH',    '#e67e22')
    if score > 2.0: return ('MEDIUM',  '#f39c12')
    return           ('ROUTINE', '#27ae60')
