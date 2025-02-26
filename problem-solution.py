import numpy as np
#لو هشتغل علي اي مشكلة زي ال XOR  هحتاج اتنين انبوت وواحد اوتبوت
input_size = 2
hidden_size = 3
output_size = 1

# بشكل عشوائي عشان تتعلم وتتحسن
weightsih = np.random.randn(input_size, hidden_size)
biasesh = np.random.randn(hidden_size)
weightsio = np.random.randn(hidden_size, output_size)
biaseso = np.random.randn(output_size)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


Epochs = 1000
l_rate = 0.1

for epoch in range(Epochs):
    hiddeni = np.dot(X, weightsih) + biasesh
    hiddeno = 1 / (1 + np.exp(-hiddeni))


    outputi = np.dot(hiddeno, weightsio) + biaseso
    output = 1 / (1 + np.exp(-outputi))

    # mse
    loss = np.mean((y - output) ** 2)


    outpute = y - output
    outputd = outpute * (output * (1 - output))

    hiddene = np.dot(outputd, weightsio.T)
    hiddend = hiddene * (hiddeno * (1 - hiddeno))  # الاشتقاق

    weightsio += np.dot(hiddeno.T, outputd) * l_rate
    biaseso += np.sum(outputd, axis=0) * l_rate
    weightsih += np.dot(X.T, hiddend) * l_rate
    biasesh += np.sum(hiddend, axis=0) * l_rate


    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

hiddeni = np.dot(X, weightsih) + biasesh
hiddeno = 1 / (1 + np.exp(-hiddeni))
outputi = np.dot(hiddeno, weightsio) + biaseso
output = 1 / (1 + np.exp(-outputi))

print("the predict is :")
print(output)
