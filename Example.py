from kan import *
set_seed(0)

model = KAN(width=[2, 1, 1], grid=3, k=3, seed=1).speed(compile=False)

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

dataset = create_dataset(f, n_var=2, train_num=5, test_num=2)


model.fit(dataset, opt="LBFGS", steps=10)

print(model(dataset["test_input"]))

model.saveckpt("microkan/m1")

loaded = KAN.loadckpt("microkan/m1")

print(loaded(dataset["test_input"]))
