import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from myModeler_v2 import myModel
import matplotlib.pyplot as plt
import numpy as np

print(torch.cuda.is_available())

#from google.colab import drive
#drive.mount('/content/drive')


# Veri setini (MNIST - El yazısı rakamlar) indiriyoruz ve normalize ediyoruz
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



model = myModel()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Toplam Parametre: {total_params}")
print(f"Eğitilebilir Parametre: {trainable_params}")

# 1. GPU kullanılabilir mi kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 2. Modelin nerede olduğuna bak
# model.parameters() içindeki ilk ağırlığın cihazına bakarız
print(f"Model şu an şurada: {next(model.parameters()).device}")

model.to(device)

# 2. Modelin nerede olduğuna bak
# model.parameters() içindeki ilk ağırlığın cihazına bakarız
print(f"Model şu an şurada: {next(model.parameters()).device}")

# criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion.to(device)





epochs = 20 # Verinin üzerinden kaç kez geçileceği
gradients = [[],[],[],[],[]]

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 1. Gradyanları sıfırla
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 2. Forward pass: Tahmin yap
        outputs = model(images)

        # 3. Kaybı (hata oranını) hesapla
        loss = criterion(outputs, labels)

        # 4. Backward pass: Hatayı geriye yay ve ağırlıkları güncelle
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Kayıp: {running_loss/len(train_loader)}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient mean: {param.grad.abs().mean().item()}")
            gradients[0].append(param.grad.abs().mean().item()[0])
            gradients[1].append(param.grad.abs().mean().item()[2])
            gradients[2].append(param.grad.abs().mean().item()[4])
            gradients[3].append(param.grad.abs().mean().item()[6])
            gradients[4].append(param.grad.abs().mean().item()[8])

print("Eğitim Tamamlandı!")

plt.plot(np.arange(len(gradients[0])),gradients[0],label = model.named_parameters()[0])
plt.plot(np.arange(len(gradients[1])),gradients[1],label = model.named_parameters()[2])
plt.plot(np.arange(len(gradients[2])),gradients[2],label = model.named_parameters()[4])
plt.plot(np.arange(len(gradients[3])),gradients[3],label = model.named_parameters()[6])
plt.plot(np.arange(len(gradients[4])),gradients[4],label = model.named_parameters()[8])
plt.legend()
plt.show()

from modulefinder import test
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # Modeli değerlendirme moduna al (dropout vb. varsa kapatır)

    with torch.no_grad(): # Gradyan hesaplamayı kapat (hafıza tasarrufu sağlar)
        for x, y in loader:
            # Tahmin yap
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1) # En yüksek skorlu sınıfı seç

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train() # Modeli tekrar eğitim moduna döndür
    return float(num_correct) / num_samples * 100

# Kullanımı:
print(f"Test Başarımı: %{check_accuracy(test_loader, model):.2f}")


# Modelinizin isminin 'SimpleModel' olduğunu varsayalım
# Kaydedilecek dosya yolunu belirleyin (Drive'a bağlıysanız orayı kullanın)
model_path = "myModel_v2.pth"

# Model ağırlıklarını kaydetme
torch.save(model.state_dict(), model_path)
print(f"Model başarıyla şuraya kaydedildi: {model_path}")



# modeli görselleştir.
model.eval()
dummy_input = torch.randn(1, 3, 32, 32).to(device)

onnx_name = "myModel_v2.onnx"

torch.onnx.export(
    model,                      # Eğitilen model
    dummy_input,                # Örnek girdi
    onnx_name,             # Kaydedilecek dosya adı
    export_params=True,         # Eğitilmiş ağırlıkları da içine ekle
    opset_version=11,           # Standart operasyon seti sürümü
    do_constant_folding=True,   # Gereksiz işlemleri optimize et (sadeleştir)
    input_names=['input'],      # Girdi katmanının adı (opsiyonel)
    output_names=['output'],    # Çıktı katmanının adı (opsiyonel)
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Farklı batch size'lara izin ver
)

print(f"Model başarıyla {onnx_name} olarak kaydedildi!")