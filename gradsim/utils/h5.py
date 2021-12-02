# https://github.com/fab-jul/hdf5_dataloader
import datetime
import glob
import h5py
import numpy as np
import os
import pickle
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

default_opener = lambda p_: h5py.File(p_, "r")


class HDF5Dataset(Dataset):
    @staticmethod
    def _get_num_in_shard(shard_p, opener=default_opener):
        base_dir = os.path.dirname(shard_p)
        p_to_num_per_shard_p = os.path.join(base_dir, "num_per_shard.pkl")
        # Speeds up filtering massively on slow file systems...
        if os.path.isfile(p_to_num_per_shard_p):
            with open(p_to_num_per_shard_p, "rb") as f:
                p_to_num_per_shard = pickle.load(f)
                num_per_shard = p_to_num_per_shard[os.path.basename(shard_p)]
        else:
            # print(f'\rh5: Opening {shard_p}... ', end='')
            try:
                with opener(shard_p) as f:
                    num_per_shard = len([key for key in list(f.keys()) if key != "len"])
            except:
                print(f"h5: Could not open {shard_p}!")
                num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lenghts(file_ps, opener=default_opener, remove_last_hdf5=False):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_ps: list of .hdf5 files
        :param opener:
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        file_ps = sorted(file_ps)  # we assume that smallest shard is at the end
        if remove_last_hdf5:
            file_ps = file_ps[:-1]
        num_per_shard_prev = None
        ps = []
        for i, p in enumerate(file_ps):
            num_per_shard = HDF5Dataset._get_num_in_shard(p, opener)
            if num_per_shard == -1:
                continue
            if num_per_shard_prev is None:  # first file
                num_per_shard_prev = num_per_shard
                ps.append(p)
                continue
            if num_per_shard_prev < num_per_shard:
                raise ValueError(
                    "Expected all shards to have the same number of elements,"
                    f"except last one. Previous had {num_per_shard_prev} elements, current ({p}) has {num_per_shard}!"
                )
            if num_per_shard_prev > num_per_shard:  # assuming this is the last
                is_last = i == len(file_ps) - 1
                if not is_last:
                    raise ValueError(
                        f"Found shard with too few elements, and it is not the last one! {p}\n"
                        f"Last: {file_ps[-1]}\n"
                        f"Make sure to sort file_ps before filtering."
                    )
                print(f"Last shard: {p}, has {num_per_shard} elements...")
            # else: # same numer as before, all good
            ps.append(p)
        assert num_per_shard_prev is not None
        return (
            ps,
            num_per_shard_prev,
            (len(ps) - 1) * num_per_shard_prev + num_per_shard,
        )

    def __init__(
        self,
        data_dir,
        read_only_seqs=False,
        remove_last_hdf5=False,
        skip_shards=0,
        shuffle_shards=False,
        opener=default_opener,
        seed=29,
    ):
        self.data_dir = data_dir
        self.read_only_seqs = read_only_seqs
        self.remove_last_hdf5 = remove_last_hdf5
        self.skip_shards = skip_shards
        self.shuffle_shards = shuffle_shards
        self.opener = opener
        self.seed = seed

        self.shard_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.shard_paths) > 0, (
            "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir
        )

        (
            self.shard_ps,
            self.num_per_shard,
            self.total_num,
        ) = HDF5Dataset.check_shard_lenghts(
            self.shard_paths, self.opener, self.remove_last_hdf5
        )

        # Skip shards
        assert self.skip_shards < len(self.shard_ps), (
            "h5: Cannot skip all shards! Found "
            + str(len(self.shard_ps))
            + " shards in "
            + self.data_dir
            + " ; len(self.shard_paths) = "
            + str(len(self.shard_paths))
            + "; remove_last_hdf5 = "
            + str(self.remove_last_hdf5)
        )
        self.shard_ps = self.shard_ps[self.skip_shards :]
        self.total_num -= self.skip_shards * self.num_per_shard

        assert len(self.shard_ps) > 0, (
            "h5: Could not find .hdf5 files! Dir: "
            + self.data_dir
            + " ; len(self.shard_paths) = "
            + str(len(self.shard_paths))
            + "; remove_last_hdf5 = "
            + str(self.remove_last_hdf5)
        )

        self.num_of_shards = len(self.shard_ps)

        # print("Loaded hdf5 shards in", self.data_dir)
        # print("h5: paths", len(self.shard_ps), "; num_per_shard", self.num_per_shard, "; total", self.total_num)

        # Shuffle shards
        if self.shuffle_shards:
            np.random.seed(seed)
            if self.total_num != self.num_per_shard * self.num_of_shards:
                ps = self.shard_ps[:-1]
                np.random.shuffle(ps)
                self.shard_ps = ps + [self.shard_ps[-1]]
            else:
                np.random.shuffle(self.shard_ps)

    def __len__(self):
        return self.total_num

    # def __getitem__(self, index):
    #     idx = index % self.total_num
    #     shard_idx = idx // self.num_per_shard
    #     idx_in_shard = str(idx % self.num_per_shard)
    #     # Read from shard
    #     with self.opener(self.shard_ps[shard_idx]) as f:
    #         data = torch.from_numpy(f[idx_in_shard][()]).float()
    #     return data

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx = idx // self.num_per_shard
        idx_in_shard = idx % self.num_per_shard
        # # Read from shard
        # with self.opener(self.shard_ps[shard_idx]) as f:
        #     data = torch.from_numpy(f[idx_in_shard]).float()
        return shard_idx, idx_in_shard

    def retrieve_data(self, f, data_name, i):
        if i == 0:
            return f[data_name][i]
        else:
            try:
                data = f[data_name][i]
            except:
                data = f[data_name][:][0]
            return data

    def HDF5_collate_fn(self, batch):
        shard_idxs, idxs_in_shard = list(zip(*batch))
        idxs = np.stack([shard_idxs, idxs_in_shard]).T
        # Sort
        idx_args = np.core.records.fromarrays(
            [idxs[:, 0], idxs[:, 1]], names="a, b"
        ).argsort()
        rev_idx_args = np.argsort(idx_args)
        sorted_idxs = idxs[idx_args]
        # Retrieve data
        vids = []
        if not self.read_only_seqs:
            shape = []
            position = []
            orientation = []
            mass = []
            fric = []
            elas = []
            color = []
            scale = []
            force_application_points = []
            force_magnitude = []
            force_direction = []
            linear_velocity = []
            angular_velocity = []
        for shard_idx in np.unique(sorted_idxs[:, 0]):
            idx_in_shard = sorted_idxs[:, 1][sorted_idxs[:, 0] == shard_idx]
            # Read from shard
            with self.opener(self.shard_ps[shard_idx]) as f:
                for i in idx_in_shard:
                    vids.append(f["sequence"][i])
                    if not self.read_only_seqs:
                        shape.append(self.retrieve_data(f, "shape", i))
                        position.append(self.retrieve_data(f, "position", i))
                        orientation.append(self.retrieve_data(f, "orientation", i))
                        mass.append(self.retrieve_data(f, "mass", i))
                        fric.append(self.retrieve_data(f, "fric", i))
                        elas.append(self.retrieve_data(f, "elas", i))
                        color.append(self.retrieve_data(f, "color", i))
                        scale.append(self.retrieve_data(f, "scale", i))
                        force_application_points.append(self.retrieve_data(f, "force_application_points", i))
                        force_magnitude.append(self.retrieve_data(f, "force_magnitude", i))
                        force_direction.append(self.retrieve_data(f, "force_direction", i))
                        linear_velocity.append(self.retrieve_data(f, "linear_velocity", i))
                        angular_velocity.append(self.retrieve_data(f, "angular_velocity", i))
        st = lambda a: np.stack(a)[rev_idx_args]
        if self.read_only_seqs:
            return st(vids)
        else:
            return (
                st(vids),
                st(shape),
                st(position),
                st(orientation),
                st(mass),
                st(fric),
                st(elas),
                st(color),
                st(scale),
                st(force_application_points),
                st(force_magnitude),
                st(force_direction),
                st(linear_velocity),
                st(angular_velocity),
            )


class HDF5Maker:
    def __init__(
        self,
        out_dir,
        num_per_shard=1000,
        max_shards=None,
        name_fmt="shard_{:04d}.hdf5",
        force=False,
        compression="gzip",
    ):

        self.out_dir = out_dir
        self.num_per_shard = num_per_shard
        self.max_shards = max_shards
        self.name_fmt = name_fmt
        self.force = force
        self.compression = compression

        if os.path.isdir(self.out_dir):
            if not self.force:
                raise ValueError(f"{self.out_dir} already exists.")
            print(f"Removing *.hdf5 files from {self.out_dir}...")
            files = glob.glob(os.path.join(self.out_dir, "*.hdf5"))
            for file in files:
                os.remove(file)
        else:
            os.makedirs(self.out_dir)

        self.log_file = open(os.path.join(self.out_dir, "log.txt"), "wt")
        log = "\n".join(
            "{}={}".format(k, v)
            for k, v in [
                ("out_dir", self.out_dir),
                ("num_per_shard", self.num_per_shard),
                ("max_shards", self.max_shards),
                ("name_fmt", self.name_fmt),
                ("force", self.force),
                ("compression", self.compression),
            ]
        )
        self.logg(log)

        self.writer = None
        self.shard_paths = []
        self.shard_number = 0

        # To save num_of_objs in each item
        shard_idx = 0
        idx_in_shard = 0

        self.create_new_shard()

    def create_new_shard(self):

        if self.writer:
            self.writer.close()

        self.shard_number += 1

        if self.max_shards is not None and self.shard_number == self.max_shards + 1:
            log = f"Created {self.max_shards} shards, ENDING."
            self.logg(log)
            return

        self.shard_p = os.path.join(
            self.out_dir, self.name_fmt.format(self.shard_number)
        )
        assert not os.path.exists(
            self.shard_p
        ), f"Record already exists! {self.shard_p}"
        self.shard_paths.append(self.shard_p)

        log = f"{datetime.datetime.now():%Y%m%d_%H%M%S} Creating shard # {self.shard_number}: {self.shard_p}..."
        self.logg(log)
        self.writer = h5py.File(self.shard_p, "w")

        self.sequence = None

    def add_data_to_dataset(self, dataset, data):
        dataset.resize(dataset.len() + 1, axis=0)
        dataset[dataset.len() - 1] = data

    def add_data(
        self,
        sequence,
        shape,
        position,
        orientation,
        mass,
        fric,
        elas,
        color,
        scale,
        force_application_points,
        force_magnitude,
        force_direction,
        linear_velocity,
        angular_velocity,
        random_force_magnitude,
        random_force_direction,
        random_linear_velocity,
        random_angular_velocity,
        objects,
        camera_distance=12.0,
        elevation=30.0,
        azimuth=0.0,
    ):

        if self.sequence is None:
            # sequence
            self.sequence = self.writer.create_dataset(
                "sequence",
                data=sequence[None, ...],
                maxshape=(None, *sequence.shape),
                compression=self.compression,
                dtype=np.uint8,
                chunks=True,
            )
            # shape
            self.shape = self.writer.create_dataset(
                "shape",
                data=np.array(shape)[None, ...],
                maxshape=(None,),
                compression=self.compression,
                dtype=np.int,
                chunks=True,
            )
            # position
            self.position = self.writer.create_dataset(
                "position",
                data=np.array(position)[None, ...],
                maxshape=(None, *np.array(position).shape),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # orientation
            self.orientation = self.writer.create_dataset(
                "orientation",
                data=np.array(orientation)[None, ...],
                maxshape=(None, *np.array(orientation).shape),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # mass
            self.mass = self.writer.create_dataset(
                "mass",
                data=np.array(mass)[None, ...],
                maxshape=(None,),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # fric
            self.fric = self.writer.create_dataset(
                "fric",
                data=np.array(fric)[None, ...],
                maxshape=(None,),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # elas
            self.elas = self.writer.create_dataset(
                "elas",
                data=np.array(elas)[None, ...],
                maxshape=(None,),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # color
            self.color = self.writer.create_dataset(
                "color",
                data=np.array(color)[None, ...],
                maxshape=(None, 3),
                compression=self.compression,
                dtype=np.uint8,
                chunks=True,
            )
            # scale
            self.scale = self.writer.create_dataset(
                "scale",
                data=np.array(scale)[None, ...],
                maxshape=(None,),
                compression=self.compression,
                dtype=np.float,
                chunks=True,
            )
            # force_application_points
            self.force_application_points = self.writer.create_dataset(
                "force_application_points",
                data=np.array(force_application_points)[None, ...],
                maxshape=(None, *np.array(force_application_points).shape),
                compression=self.compression,
                dtype=int,
                chunks=True,
            )
            # random_force_magnitude
            if random_force_magnitude:
                self.force_magnitude = self.writer.create_dataset(
                    "force_magnitude",
                    data=np.array(force_magnitude)[None, ...],
                    maxshape=(None,),
                    compression=self.compression,
                    dtype=np.float,
                    chunks=True,
                )
            else:
                self.force_magnitude = self.writer.create_dataset(
                    "force_magnitude", data=np.array(force_magnitude)[None, ...],
                )
            # random_force_direction
            if random_force_direction:
                self.force_direction = self.writer.create_dataset(
                    "force_direction",
                    data=np.array(force_direction)[None, ...],
                    maxshape=(None, 3),
                    compression=self.compression,
                    dtype=np.float,
                    chunks=True,
                )
            else:
                self.force_direction = self.writer.create_dataset(
                    "force_direction", data=np.array(force_direction)[None, ...],
                )
            # linear_velocity
            if random_linear_velocity:
                self.linear_velocity = self.writer.create_dataset(
                    "linear_velocity",
                    data=np.array(linear_velocity)[None, ...],
                    maxshape=(None, *np.array(linear_velocity).shape),
                    compression=self.compression,
                    dtype=np.float,
                    chunks=True,
                )
            else:
                self.linear_velocity = self.writer.create_dataset(
                    "linear_velocity", data=np.array(linear_velocity)[None, ...],
                )
            # angular_velocity
            if random_angular_velocity:
                self.angular_velocity = self.writer.create_dataset(
                    "angular_velocity",
                    data=np.array(angular_velocity)[None, ...],
                    maxshape=(None, *np.array(angular_velocity).shape),
                    compression=self.compression,
                    dtype=np.float,
                    chunks=True,
                )
            else:
                self.angular_velocity = self.writer.create_dataset(
                    "angular_velocity", data=np.array(angular_velocity)[None, ...],
                )
            # objfiles
            self.objects = self.writer.create_dataset(
                "objects", data=[n.encode("ascii", "ignore") for n in objects], dtype='S10'
            )
            # camera
            self.camera_distance = self.writer.create_dataset(
                "camera_distance", data=camera_distance
            )
            self.elevation = self.writer.create_dataset("elevation", data=elevation)
            self.azimuth = self.writer.create_dataset("azimuth", data=azimuth)
        else:
            self.add_data_to_dataset(self.sequence, sequence)
            self.add_data_to_dataset(self.shape, shape)
            self.add_data_to_dataset(self.position, position)
            self.add_data_to_dataset(self.orientation, orientation)
            self.add_data_to_dataset(self.mass, mass)
            self.add_data_to_dataset(self.fric, fric)
            self.add_data_to_dataset(self.elas, elas)
            self.add_data_to_dataset(self.color, color)
            self.add_data_to_dataset(self.scale, scale)
            self.add_data_to_dataset(
                self.force_application_points, force_application_points
            )
            if random_force_magnitude:
                self.add_data_to_dataset(
                    self.force_magnitude, np.array(force_magnitude)
                )
            if random_force_direction:
                self.add_data_to_dataset(
                    self.force_direction, np.array(force_direction)
                )
            if random_linear_velocity:
                self.add_data_to_dataset(
                    self.linear_velocity, np.array(linear_velocity)
                )
            if random_angular_velocity:
                self.add_data_to_dataset(
                    self.angular_velocity, np.array(angular_velocity)
                )

        if self.sequence.len() == self.num_per_shard:
            self.create_new_shard()

    def close(self):

        self.writer.close()
        assert len(self.shard_paths)

        # Writing num_per_shard.pkl
        p_to_num_per_shard = {
            os.path.basename(shard_p): self.num_per_shard
            for shard_p in self.shard_paths
        }
        last_shard_p = self.shard_paths[-1]
        try:
            with h5py.File(last_shard_p, "r") as f:
                p_to_num_per_shard[os.path.basename(last_shard_p)] = f["sequence"].len()
        except KeyError:
            os.remove(last_shard_p)
            self.shard_paths = self.shard_paths[:-1]
            del p_to_num_per_shard[os.path.basename(last_shard_p)]

        log = f"{datetime.datetime.now():%Y%m%d_%H%M%S} Writing {os.path.join(self.out_dir, 'num_per_shard.pkl')}"
        self.logg(log)
        self.logg(str(p_to_num_per_shard))
        with open(os.path.join(self.out_dir, "num_per_shard.pkl"), "wb") as f:
            pickle.dump(p_to_num_per_shard, f)

    def logg(self, log):
        print(log)
        self.log_file.write(log)
        self.log_file.write("\n")
        self.log_file.flush()


if __name__ == "__main__":

    # Make
    a = (torch.randn(20, 120, 256, 256, 4) * 20).type(torch.uint8)
    h5_maker = HDF5Maker(
        "/home/voletiv/EXPERIMENTS/h5", num_per_shard=10, force=True, compression="gzip"
    )
    for data in a:
        h5_maker.add_data(data)

    h5_maker.close()

    # Read
    h5_ds = HDF5Dataset("/home/voletiv/EXPERIMENTS/h5", remove_last_hdf5=True)
    data = h5_ds[0]
    assert data == (0, 0)

    # Dataloader
    h5_dl = DataLoader(h5_ds, batch_size=4, shuffle=False)
    data = next(iter(h5_dl))
    assert torch.all(
        torch.stack(data)
        == torch.stack([torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 2, 3])])
    )

    # DataLoader with collate
    h5_dl = DataLoader(
        h5_ds, batch_size=4, shuffle=False, collate_fn=h5_ds.HDF5_collate_fn
    )
    data = next(iter(h5_dl))

    assert torch.all(data == torch.stack(a[:4]))

    # Dataset with last, DataLoader with collate
    h5_ds = HDF5Dataset("/home/voletiv/EXPERIMENTS/h5", remove_last_hdf5=False)
    h5_dl = DataLoader(
        h5_ds, batch_size=7, shuffle=True, collate_fn=h5_ds.HDF5_collate_fn
    )
    data = []
    for d in tqdm(h5_dl):
        data.append(d)
