import numpy as np

data = np.load("Town10_seq_0001_full_review/cameras/front/depth/0000025709.npz")

print("Keys:", list(data.keys()))

for key in data.keys():
    arr = data[key]
    print(f"\n[{key}]")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    print(f"  Min   : {arr.min():.4f}")
    print(f"  Max   : {arr.max():.4f}")
    print(f"  Mean  : {arr.mean():.4f}")
    print(f"  Sample values (flat[:10]): {arr.flatten()[:10]}")

"""Keys: ['depth']

[depth]
  Shape : (600, 800)
  Dtype : float16
  Min   : 3.2109
  Max   : 1000.0000
  Mean  : 247.5000
  Sample values (flat[:10]): [1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000.]"""