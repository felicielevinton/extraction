import spikeinterface.sorters as ss
import spikeinterface.extractors as se

file_path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240410_SESSION_04/ephys.rhd'      

test_recording, _ = se.toy_example(
        duration=30,
        seed=0,
        num_channels=64,
        num_segments=1
    )

recording = se.read_intan("file_path", stream_id=None, stream_name=None, all_annotations=False)

sorting = ss.run_kilosort3(
        recording=test_recording,
        output_folder="kilosort3",
        singularity_image=False,
        docker_image=True
    )
print(sorting)