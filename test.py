import torch

# Example mel-spectrogram matrix of shape (T, d)
melspec = torch.tensor([
    [0.0, 0.1, 0.2],
    [0.0, 0.0, 0.0],  # All-zero frame
    [0.1, 0.0, 0.2],
    [0.1, 0.0, 0.2]
])

print(melspec.shape)

# Check for any all-zero frames (rows)
zero_frames = torch.all(melspec == 0, dim=1)
has_zero_frame = torch.any(zero_frames).item()

# Check for any all-zero features (columns)
zero_features = torch.all(melspec == 0, dim=0)
has_zero_feature = torch.any(zero_features).item()

print(f'Any all-zero frames: {has_zero_frame}')  # True if there's any all-zero frame
print(f'Any all-zero features: {has_zero_feature}')  # True if there's any all-zero feature
