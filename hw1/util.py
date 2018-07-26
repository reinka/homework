from sklearn.preprocessing import StandardScaler


def train_test_scale(trX, teX, trY, teY):
    scalerX, scalerY = StandardScaler().fit(trX), StandardScaler().fit(trY)

    trX = scalerX.transform(trX)
    teX = scalerX.transform(teX)

    trY = scalerY.transform(trY)
    teY = scalerY.transform(teY)

    return trX, teX, trY, teY, scalerX, scalerY


def policy_fn(model, obs):
    if model.scalerX and model.scalerY:
        action = model.predict(model.scalerX.transform(obs))
        return model.scalerY.inverse_transform(action)

    return model.predict(obs)


def plot_loss(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
