
def normalize_filter(tensor):
    # Tensorun min ve max değerlerini bul
    t_min = tensor.min()
    t_max = tensor.max()
    eps = 1e-8  

    return (tensor - t_min) / (t_max - t_min + eps)

def find_close_zeros(tensor, threshold=0.01):
    
    count = (tensor.abs() < threshold).sum().item()

    return count / tensor.numel() * 100