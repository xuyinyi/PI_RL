import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, net_params):
        super(MLPModel, self).__init__()

        if net_params["active_function"] == 1:
            self.active = nn.LeakyReLU()
        elif net_params["active_function"] == 0:
            self.active = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {net_params['active_function']}")

        self.dropout = nn.Dropout(net_params["dropout"])
        self.n_hidden = net_params["num_hidden"]
        self.hidden_dim = net_params["hidden_dim"]
        self.layers = nn.ModuleList()
        # first hidden layer
        self.layers.append(nn.Sequential(
            nn.Linear(net_params["input_dim"], self.hidden_dim),
            self.active))
        for _ in range(self.n_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.active))

        if net_params['sigmoid']:
            self.layers.append(nn.Sequential(self.dropout, nn.Linear(self.hidden_dim, 1), nn.Sigmoid()))
        else:
            self.layers.append(nn.Sequential(self.dropout, nn.Linear(self.hidden_dim, 1)))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, inputs):
        output = self.layers[0](inputs)
        for _ in range(self.n_hidden):
            output = self.layers[_ + 1](output)
        output = self.layers[-1](output)
        return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss
