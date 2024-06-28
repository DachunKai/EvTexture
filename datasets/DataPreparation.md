## Symlink/Put all the datasets here

It is recommended to symlink your dataset root to this folder - `datasets` with the command `ln -s xxx yyy`.

## Prepare Datasets
We regroup the dataset into [HDF5](https://www.h5py.org/) format because it offers better read IO performance.

### Simulating Events and get voxel grids
* **Step 1: Event data generation.**

	We follow [vid2e](https://github.com/uzh-rpg/rpg_vid2e) to simulate [REDS](https://seungjunnah.github.io/Datasets/reds.html), [Viemo-90K](https://github.com/anchen1011/toflow) and [Vid4](https://mmagic.readthedocs.io/en/stable/dataset_zoo/vid4.html) events in high-resolution. Notice that vid2e repo use pretrained [FILM](https://github.com/google-research/frame-interpolation) video frame interpolation model to firstly interpolate frames, where we use pretrained [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) to interpolate frames. Then we use `esim_py` Pypi package in [vid2e](https://github.com/uzh-rpg/rpg_vid2e) to simulate events from interpolated sequences. Our simulator parameters configuration is as follows:
    ```python
	import random
	import esim_py
	config = {
		'refractory_period': 1e-4,
		'CT_range': [0.05, 0.5],
		'max_CT': 0.5,
		'min_CT': 0.02,
		'mu': 1,
		'sigma': 0.1,
		'H': clip.height,
		'W': clip.width,
		'log_eps': 1e-3,
		'use_log': True,
	}

	Cp = random.uniform(config['CT_range'][0], config['CT_range'][1])
	Cn = random.gauss(config['mu'], config['sigma']) * Cp
	Cp = min(max(Cp, config['min_CT']), config['max_CT'])
	Cn = min(max(Cn, config['min_CT']), config['max_CT'])
	esim = esim_py.EventSimulator(Cp,
								Cn,
								config['refractory_period'],
								config['log_eps'],
								config['use_log'])
	events = esim.generateFromFolder(image_folder, timestamps_file) # Generate events with shape [N, 4]
    ```
	Here, timestamps_file is user-defined. For videos with known frame rates, this file contains [0, 1.0/fps, 2.0/fps, ...]. For unknown frame rates, we assume fps = 25.0. Similar event camera simulators include [ESIM](https://github.com/uzh-rpg/rpg_esim), [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter), or [V2E](https://github.com/SensorsINI/v2e). You can also try them.

* **Step 2: Convert events to voxel grids.**

    - Refer to [events_contrast_maximization](https://github.com/TimoStoff/events_contrast_maximization/blob/master/tools/event_packagers.py) for creating the [hdf5](https://docs.h5py.org/en/stable/) data structure to accerate IO processing.
	- Then convert events to voxel grids following [events_to_voxel_torch](https://github.com/TimoStoff/event_utils/blob/master/lib/representations/voxel_grid.py#L114-L153) (we set B=5).

* **Step 3: Generate backward voxel grids to suit our bidirectional network.**
	```python
	if backward:
		xs = torch.flip(xs, dims=[0])
		ys = torch.flip(ys, dims=[0])
		ts = torch.flip(t_end - ts + t_start, dims=[0]) # t_end and t_start represent the timestamp range of the events to be flipped, typically the timestamps of two consecutive frames.
		ps = torch.flip(-ps, dims=[0])
	voxel = events_to_voxel_torch(xs, ys, ts, ps, bins, device=None, sensor_size=sensor_size)
	```

* **Step 4: Voxel normalization.**
	```python
	def voxel_normalization(voxel):
    """
        normalize the voxel same as https://arxiv.org/abs/1912.01584 Section 3.1
        Params:
            voxel: torch.Tensor, shape is [num_bins, H, W]

        return:
            normalized voxel
    """
		# check if voxel all element is 0
		a,b,c = voxel.shape
		tmp = torch.zeros(a, b, c)
		if torch.equal(voxel, tmp):
			return voxel
		abs_voxel, _ = torch.sort(torch.abs(voxel).view(-1, 1).squeeze(1))
		first_non_zero_idx = torch.nonzero(abs_voxel)[0].item()
		non_zero_voxel = abs_voxel[first_non_zero_idx:]
		norm_idx = math.floor(non_zero_voxel.shape[0] * 0.98)
		ones = torch.ones_like(voxel)
		normed_voxel = torch.where(torch.abs(voxel) < non_zero_voxel[norm_idx], voxel / non_zero_voxel[norm_idx], voxel)
		normed_voxel = torch.where(normed_voxel >= non_zero_voxel[norm_idx], ones, normed_voxel)
		normed_voxel = torch.where(normed_voxel <= -non_zero_voxel[norm_idx], -ones, normed_voxel)
		return normed_voxel
	```

* **Step 5: Downsample voxels.**
	Apply bicubic downsample using [torch.nn.functional.interpolate](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) to converted event voxels to generate low-resolution event voxel.

	**[Note]**: If you are only inferring on your own low-resolution video, there is no need to downsample the event voxels.

### Dataset Structure
* Training set
	* [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset. The meta info files are [meta_info_REDS_h5_train.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_REDS_h5_train.txt) and [meta_info_REDS_h5_test.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_REDS_h5_test.txt). Prepare REDS_h5 structure be:
	```arduino
	REDS_h5
	├── HR
	│   ├── train
	│   │   ├── 001.h5
	│   │   ├── ...
	│   ├── test
	│       ├── 000.h5
	│       ├── ...
	├── LRx4
	│   ├── train
	│   │   ├── 001.h5
	│   │   ├── ...
	│   ├── test
	│       ├── 000.h5
	│       ├── ...
	```
	* [Viemo-90K](https://github.com/anchen1011/toflow) dataset. The meta info files are [meta_info_Vimeo_h5_train.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_Vimeo_h5_train.txt) and [meta_info_Vimeo_h5_test.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_Vimeo_h5_test.txt). Prepare Vimeo_h5 structure be:
	```arduino
	Vimeo_h5
	├── HR
	│   ├── train
	│   │   ├── 00001_0001.h5
	│   │   ├── ...
	│   ├── test
	│       ├── 00001_0266.h5
	│       ├── ...
	├── LRx4
	│   ├── train
	│   │   ├── 00001_0001.h5
	│   │   ├── ...
	│   ├── test
	│       ├── 00001_0266.h5
	│       ├── ...
	```
	* [CED](https://rpg.ifi.uzh.ch/CED.html) dataset. The meta info files are [meta_info_CED_h5_train.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_CED_h5_train.txt) and [meta_info_CED_h5_test.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_CED_h5_test.txt). Prepare CED_h5 structure be:
	```arduino
	├────CED_h5
	│   ├────HR
	│   │   ├────train
	│   │   │   ├────calib_fluorescent.h5
	│   │   │   ├────...
	│   │   ├────test
	│   │       ├────indoors_foosball_2.h5
	│   │       ├────...
	│   ├────LRx2
	│   │   ├────train
	│   │   │   ├────calib_fluorescent.h5
	│   │   │   ├────...
	│   │   ├────test
	│   │       ├────indoors_foosball_2.h5
	│   │       ├────...
	│   ├────LRx4
	│       ├────train
	│       │   ├────calib_fluorescent.h5
	│       │   ├────...
	│       ├────test
	│           ├────indoors_foosball_2.h5
	│           ├────...
	```

* Testing set
	* [REDS4](https://seungjunnah.github.io/Datasets/reds.html) dataset.
    * [Vimeo-90K-T](https://github.com/anchen1011/toflow) dataset.
    * [Vid4](https://mmagic.readthedocs.io/en/stable/dataset_zoo/vid4.html) dataset. The meta info file is [meta_info_Vid4_h5_test.txt](https://github.com/DachunKai/EvTexture/blob/main/basicsr/data/meta_info/meta_info_Vid4_h5_test.txt). Prepare Vid4_h5 structure be:
	```arduino
	Vid4_h5
	├── HR
	│   ├── test
	│   │   ├── calendar.h5
	│   │   ├── ...
	├── LRx4
	│   ├── test
	│       ├── calendar.h5
	│       ├── ...
	```
    * [CED](https://rpg.ifi.uzh.ch/CED.html) dataset.

* Hdf5 file example

   We show our HDF5 file structure using the `calendar.h5` file from the Vid4 dataset.

	```arduino
	calendar.h5
	├── images
	│   ├── 000000 # frame, ndarray, [H, W, C]
	│   ├── ...
	├── voxels_f
	│   ├── 000000 # forward event voxel, ndarray, [Bins, H, W]
	│   ├── ...
	├── voxels_b
	│   ├── 000000 # backward event voxel, ndarray, [Bins, H, W]
	│   ├── ...
	```