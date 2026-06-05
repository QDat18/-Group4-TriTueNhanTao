import math
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    """
    Tạo bộ lập lịch học tập (learning rate scheduler) kết hợp Warmup tuyến tính 
    và Cosine Annealing Decay.
    
    Trong suốt giai đoạn warmup_epochs đầu tiên, lr tăng tuyến tính từ min_lr lên base_lr.
    Sau đó, lr giảm dần về min_lr theo đường cong cosine cho đến total_epochs.
    
    Args:
        optimizer: Bộ tối ưu hóa (optimizer) của PyTorch.
        warmup_epochs: Số epoch chạy warmup.
        total_epochs: Tổng số epoch training.
        base_lr: Tốc độ học cơ bản cao nhất đạt được sau warmup.
        min_lr: Tốc độ học tối thiểu ở điểm cuối của decay.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Giai đoạn Linear Warmup: tăng dần từ min_lr lên base_lr
            ratio = min_lr / base_lr
            return ratio + (1.0 - ratio) * (current_epoch / warmup_epochs)
        else:
            # Giai đoạn Cosine Annealing: giảm dần từ base_lr về min_lr
            progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = min_lr / base_lr
            return ratio + (1.0 - ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
