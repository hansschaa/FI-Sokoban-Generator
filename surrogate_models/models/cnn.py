import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    """Bloque CNN básico sin conexiones residuales ni atención."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        return out

class SokobanCNNRegressor(nn.Module):
    """
    CNN Básica para Regresión (Ablation Study).
    Arquitectura idéntica a la SE-ResNet pero usando bloques CNN estandar.
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(CNNBlock(32,  64, stride=1),  CNNBlock(64,  64))
        self.layer2 = nn.Sequential(CNNBlock(64,  128, stride=2), CNNBlock(128, 128))
        self.layer3 = nn.Sequential(CNNBlock(128, 256, stride=2), CNNBlock(256, 256))
        self.layer4 = nn.Sequential(CNNBlock(256, 512, stride=2), CNNBlock(512, 512))
        
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


class SokobanCNNClassifier(nn.Module):
    """
    CNN Básica para Clasificación (Ablation Study).
    Arquitectura idéntica a la SE-ResNet pero usando bloques CNN estandar.
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(CNNBlock(32,  64, stride=1),  CNNBlock(64,  64))
        self.layer2 = nn.Sequential(CNNBlock(64,  128, stride=2), CNNBlock(128, 128))
        self.layer3 = nn.Sequential(CNNBlock(128, 256, stride=2), CNNBlock(256, 256))
        self.layer4 = nn.Sequential(CNNBlock(256, 512, stride=2), CNNBlock(512, 512))
        
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
