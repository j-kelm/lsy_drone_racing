import os
import h5py

folder = 'output/workers/'
output = 'output/merged.hdf5'

if __name__ == '__main__':
    with h5py.File(output, 'w', libver='latest') as out_file:
        for file in os.listdir(folder):
            if file.endswith(".hdf5"):
                with h5py.File(os.path.join(folder, file), 'r', libver='latest') as current_file:
                    # loop over tracks in other file
                    for track_key in current_file:
                        track_grp = current_file[track_key]
                        if track_key not in out_file:
                            out_file.copy(source=track_grp, dest=out_file, name=track_key)
                        elif 'track_' in track_key:
                            for worker_key in track_grp:
                                if 'worker_' in worker_key:
                                    worker_grp = track_grp[worker_key]

                                    if worker_key not in out_file[track_key]:
                                        out_file.copy(source=worker_grp, dest=out_file[track_key], name=worker_key)
