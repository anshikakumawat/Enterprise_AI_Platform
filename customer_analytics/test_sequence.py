from sequence_models import build_lstm
from sequence_models import build_gru

lstm_model = build_lstm((10, 5))
gru_model = build_gru((10, 5))

print("LSTM Model")
lstm_model.summary()

print("\nGRU Model")
gru_model.summary()