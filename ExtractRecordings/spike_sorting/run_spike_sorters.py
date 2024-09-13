import spikeinterface.sorters as ss
import spikeinterface.extractors as se


if __name__ == "__main__":
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
    # ss.run_sorter()

