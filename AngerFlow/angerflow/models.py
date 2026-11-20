class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self,layer):
        self.layers.append(layer)

    def forward(self,x,training=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out,training=training)
            return out

    def predict(self,x):
        return self.forward(x, training=False)
    