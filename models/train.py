import numpy as np
from optimizers import Adam
class NNTrainer:
    def __init__(self, layers, loss_function,
                 x_train, y_train, optimizer, x_valid=None, y_valid=None, 
                 epochs=15, batch_size=32):
        """
        - `layers`: List các lớp theo thứ tự forward
        - `loss_function`: Hàm loss
        - `optimizer`: Optimizer (chứa learning_rate)
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.epochs = epochs
        self.batch_size = batch_size

    def forwards(self, x):
        """Chạy dữ liệu qua tất cả các layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backwards(self, dL):
        """Lan truyền ngược qua tất cả các layers (đảo ngược thứ tự)"""
        for layer in reversed(self.layers):
            if isinstance(self.optimizer, Adam):
                dL = layer.backward(dL, self.optimizer)
            else:
                dL = layer.backward(dL)

    def fit(self):
      print("Training...")

      train_losses = []
      train_accuracies = []
      val_losses = []
      val_accuracies = []

      last_val_loss = 0  # Giá trị mặc định ban đầu
      last_val_accuracy = 0

      for epoch in range(self.epochs):
          epoch_loss, correct = [], 0

          for i in range(len(self.x_train)):
              image_shape = self.x_train.shape[1:]
              x = self.x_train[i].reshape((1,) + image_shape)
              y = self.y_train[i]

              # Forward
              out = self.forwards(x)
              loss = self.loss_function(out, y)
              epoch_loss.append(loss)
              correct += int(np.argmax(out) == np.argmax(y))

              # Backward
              dL = out - y
              self.backwards(dL)

          # Logging & lưu lại giá trị
          avg_loss = np.mean(epoch_loss)
          accuracy = correct / len(self.x_train)

          train_losses.append(avg_loss)
          train_accuracies.append(accuracy)

          print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

          # Validation mỗi 3 epochs
          if self.x_valid is not None and (epoch + 1) % 3 == 0:
              last_val_loss, last_val_accuracy = self.evaluate(self.x_valid, self.y_valid)

          val_losses.append(last_val_loss)
          val_accuracies.append(last_val_accuracy)

      return train_losses, train_accuracies, val_losses, val_accuracies

    
    def evaluate(self, x, y):
        val_loss, val_correct = [], 0
        image_shape = x.shape[1:]

        for i in range(len(x)):
            x_sample = x[i].reshape((1,) + image_shape)
            y_sample = y[i]

            out = self.forwards(x_sample)
            val_loss.append(self.loss_function(out, y_sample))
            val_correct += int(np.argmax(out) == np.argmax(y_sample))

        avg_val_loss = np.mean(val_loss)
        avg_val_accuracy = val_correct / len(x)

        print(f"--> Evaluation: Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy * 100:.2f}%")
        return avg_val_loss, avg_val_accuracy

    def predict(self, x):
        out = self.forwards(x)
        return np.argmax(out)
    

