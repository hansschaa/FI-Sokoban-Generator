"""
resnet.py
---------
Arquitectura ResNet Multi-Head para los Surrogate Models de Sokoban.

Dos cabezas de salida:
  - head_pushes:    predice pushes normalizados (regresión)
  - head_branching: predice branching_effective normalizado (regresión)

Anti-overfit: BatchNorm + Dropout(0.4) en el cuello compartido.
Heurística no negativa: ReLU al final de cada cabeza.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SokobanResNetRegressor(nn.Module):
    """
    Entrada : (B, 5, 25, 25)
    Salidas : pushes_pred (B,), branching_pred (B,)  — en espacio Z-score
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32,  64),  BasicBlock(64,  64))
        self.layer2 = nn.Sequential(BasicBlock(64,  128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
        )
        self.head_pushes    = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.neck(x)
        return self.head_pushes(x).squeeze(1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for Channel Attention."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """BasicBlock with Squeeze-and-Excitation mechanism."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class SokobanSEResNetRegressor(nn.Module):
    """
    Entrada : (B, 5, 25, 25)
    Salida  : pushes_pred (B,) 
    Arquitectura más profunda y con atención espacial (SE).
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        # 4 layers para más capacidad
        self.layer1 = nn.Sequential(SEBasicBlock(32,  64),  SEBasicBlock(64,  64))
        self.layer2 = nn.Sequential(SEBasicBlock(64,  128), SEBasicBlock(128, 128))
        self.layer3 = nn.Sequential(SEBasicBlock(128, 256), SEBasicBlock(256, 256))
        self.layer4 = nn.Sequential(SEBasicBlock(256, 512), SEBasicBlock(512, 512))
        
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
        )
        self.head_pushes = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.neck(x)
        return self.head_pushes(x).squeeze(1)

class AsymmetricHuberLoss(nn.Module):
    """Penaliza más las sobreestimaciones (alpha > 1)."""
    def __init__(self, delta: float = 1.0, alpha: float = 1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, pred, target):
        error = pred - target
        abs_err = torch.abs(error)
        huber = torch.where(abs_err <= self.delta,
                            0.5 * error**2,
                            self.delta * (abs_err - 0.5 * self.delta))
        weight = torch.where(error > 0, self.alpha * torch.ones_like(error), torch.ones_like(error))
        return (weight * huber).mean()


class MultiHeadRegressorLoss(nn.Module):
    # Ya no se usa, pero la mantenemos vacía por compatibilidad de imports en otros lados o la borramos
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanResNetRegressor().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n:,}")
    x = torch.randn(8, 5, 25, 25).to(device)
    p = model(x)
    print(f"Pushes:    {p.shape}  sample={p[:3].tolist()}")
    print("✅ Smoke test OK")

# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICADOR (Soluble vs Deadlock)
# ─────────────────────────────────────────────────────────────────────────────

class SokobanResNetClassifier(nn.Module):
    """
    Red similar al regresor, pero con una sola cabeza terminando en 1 neurona (sin activación,
    para usar con BCEWithLogitsLoss).
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32,  64),  BasicBlock(64,  64))
        self.layer2 = nn.Sequential(BasicBlock(64,  128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Dropout(p=dropout_p)
        )
        self.head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.neck(x)
        return self.head(x).squeeze(1)


class SokobanSEResNetClassifier(nn.Module):
    """
    Clasificador avanzado con arquitectura profunda (4 capas) y atención espacial (SE).
    Termina en 1 neurona (logit) para BCEWithLogitsLoss.
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(SEBasicBlock(32,  64),  SEBasicBlock(64,  64))
        self.layer2 = nn.Sequential(SEBasicBlock(64,  128), SEBasicBlock(128, 128))
        self.layer3 = nn.Sequential(SEBasicBlock(128, 256), SEBasicBlock(256, 256))
        self.layer4 = nn.Sequential(SEBasicBlock(256, 512), SEBasicBlock(512, 512))
        
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_p),
        )
        self.head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.neck(x)
        return self.head(x).squeeze(1)


class ClassifierLoss(nn.Module):
    """
    Usa BCEWithLogitsLoss con pos_weight para lidiar con el desbalance extremo 
    (muchos deadlocks, pocos solubles). 
    Si los solubles son Clase 1, y hay 4 veces más deadlocks (Clase 0), 
    pos_weight debe ser ~4.0 para penalizar fuerte clasificar mal un soluble, 
    o al revés si priorizamos castigar los falsos positivos (filtrar deadlocks estrictamente).
    """
    def __init__(self, pos_weight_val=1.0):
        super().__init__()
        # pos_weight > 1 aumenta recall de la clase 1 (solubles)
        # pos_weight < 1 aumenta recall de la clase 0 (deadlocks)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        
    def forward(self, logits, targets):
        # BCEWithLogitsLoss maneja el sigmoid internamente por estabilidad numérica
        self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
        return self.loss_fn(logits, targets)
