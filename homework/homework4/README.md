# Assignment 4 - Implement Simplified 3D Gaussian Splatting

This repository is the official implementation of [Assignment 4 - Implement Simplified 3D Gaussian Splatting](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/04_3DGS).  


## Training

To train the model, run this command:
First, we use Colmap to recover camera poses and a set of 3D points.
```python
python mvs_with_colmap.py --data_dir data/chair
```

Debug the reconstruction by running:
```
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

build your 3DGS model:
```
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

The above chair can be replaced with the lego dataset
## Results
Both datasets converge after 150 epochs, and the results of the 0th, 50th, 100th, and 150th epochs are shown below.
### chair
#### ours
![epoch0](./04_3DGS/data/chair/checkpoints/debug_images/epoch_0000/r_47.png "epoch0")

![epoch50](./04_3DGS/data/chair/checkpoints/debug_images/epoch_0050/r_47.png "epoch50")

![epoch100](./04_3DGS/data/chair/checkpoints/debug_images/epoch_0100/r_47.png "epoch100")

![epoch150](./04_3DGS/data/chair/checkpoints/debug_images/epoch_0150/r_47.png "epoch150")

![chair](./04_3DGS/data/chair/checkpoints/debug_images/debug_rendering.gif "chair")

loss = 0.0254

#### Original Gauss
<table>
  <tr>
    <td><img src="./origin_gaussian/chair/ours_30000/gt/00006.png" alt="chair00006" style="width:100%"></td>
    <td><img src="./origin_gaussian/chair/ours_30000/renders/00006.png" alt="chair00006 render" style="width:100%"></td>
  </tr>
</table>
loss = 0.0054

### lego
#### ours
![epoch0](./04_3DGS/data/lego/checkpoints/debug_images/epoch_0000/r_85.png "epoch0")

![epoch50](./04_3DGS/data/lego/checkpoints/debug_images/epoch_0050/r_85.png "epoch50")

![epoch100](./04_3DGS/data/lego/checkpoints/debug_images/epoch_0100/r_85.png "epoch100")

![epoch150](./04_3DGS/data/lego/checkpoints/debug_images/epoch_0150/r_85.png "epoch150")

![lego](./04_3DGS/data/lego/checkpoints/debug_images/debug_rendering.gif "lego")
loss = 0.0335

#### Original Gauss
<table>
  <tr>
    <td><img src="./origin_gaussian/lego/ours_30000/gt/00006.png" alt="lego00006" style="width:100%"></td>
    <td><img src="./origin_gaussian/lego/ours_30000/renders/00006.png" alt="lego00006 render" style="width:100%"></td>
  </tr>
</table>
loss = 0.0134