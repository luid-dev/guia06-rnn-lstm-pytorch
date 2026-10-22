import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Generación de Datos Sintéticos (Serie Temporal de una Onda Senoidal)
def generar_datos(n_puntos=500, seq_len=10):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)

    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i+seq_len])
        y.append(serie[i+seq_len])

    # Convertir a tensores de PyTorch con dimensiones tridimensionales (Batch, Seq_Len, Features)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # [N, seq_len, 1]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # [N, 1]
    return X, y

# 2. Definición de la Arquitectura RNN Simple
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out: [batch_size, seq_len, hidden_size]
        out, _ = self.rnn(x)
        # Extraemos exclusivamente el último estado temporal (Many-to-One)
        out_final = out[:, -1, :]
        pred = self.fc(out_final)
        return pred

# 3. Inicialización, Pérdida y Optimización
SEQ_LEN = 10
X, y = generar_datos(n_puntos=600, seq_len=SEQ_LEN)

model = SimpleRNNModel(input_size=1, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Bucle de Entrenamiento
epochs = 50
for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

print("¡Entrenamiento de la RNN Simple finalizado con éxito!")
