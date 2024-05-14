import matplotlib.pyplot as plt

epochs = list(range(1, 21))  # epoch 1부터 20까지
train_loss = [0.8514, 0.6652, 0.5037, 0.4494, 0.4018, 0.3909, 0.3842, 0.3828, 0.3766, 0.3786, 
              0.3794, 0.3771, 0.3775, 0.3756, 0.3772, 0.3809, 0.3783, 0.3737, 0.3779, 0.3808]
validation_loss = [0.7344, 0.6130, 0.5246, 0.5382, 0.5363, 0.5065, 0.5054, 0.5086, 0.5083, 0.5090, 
                   0.5152, 0.5102, 0.5170, 0.5053, 0.5184, 0.5129, 0.5059, 0.5077, 0.5162, 0.5127]
train_f1 = [0.49, 0.63, 0.78, 0.80, 0.82, 0.82, 0.83, 0.83, 0.83, 0.83,
             0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83]
validation_f1 = [0.56, 0.75, 0.78, 0.77, 0.78, 0.80, 0.80, 0.80, 0.79, 0.80, 
                 0.78, 0.80, 0.79, 0.80, 0.78, 0.80, 0.80, 0.80, 0.79, 0.80]

plt.figure(figsize=(10, 6))

# 각 선을 그립니다.
plt.plot(epochs, train_loss, label='Train Loss', color='red')
plt.plot(epochs, validation_loss, label='Validation Loss', color='blue')
plt.plot(epochs, train_f1, label='Train F1 Score', color='green')
plt.plot(epochs, validation_f1, label='Validation F1 Score', color='orange')

# 그래프 제목과 축 라벨을 추가합니다.
plt.title('Train & Validate')
plt.xlabel('Epoch')
plt.ylabel('Value')

# 범례를 추가합니다.
plt.legend()

# 그래프를 보여줍니다.
plt.show()
