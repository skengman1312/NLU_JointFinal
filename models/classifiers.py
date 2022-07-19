import torch.nn as nn


class IntentClassifier(nn.Module):
    """
    Intent classifier layer
    """

    def __init__(self, input_dim, n_intent_labels, dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, n_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotsClassifier(nn.Module):
    """
    Slots classifier layer
    """

    def __init__(self, input_dim, n_slots_labels, dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, n_slots_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
