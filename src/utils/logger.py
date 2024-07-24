import time

class TrainingLogger:
    """
    Logger to print training progress information such as speed of learning, current accuracy, and runtime.

    Attributes:
        start_time (float): The start time of training.
        last_time (float): The time at the last log.
    """
    def __init__(self) -> None:
        
        self.start_time: float = time.time()
        self.last_time: float = self.start_time

    def log(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        Logs the current epoch, loss, accuracy, and runtime.

        Args:
            epoch (int): The current epoch number.
            loss (float): The current loss value.
            accuracy (float): The current accuracy value.
        """
        current_time: float = time.time()
        runtime: float = current_time - self.start_time
        epoch_time: float = current_time - self.last_time
        self.last_time: float = current_time
        
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Time per epoch: {epoch_time:.2f}s | Total runtime: {runtime:.2f}s")