"""R15g: reading VisDrone2019-VID annotations and merging tile detections."""

import io
import zipfile

import numpy as np

from genetic_coil_arena.experiments import r15g_video as rg


def test_parse_vid_keeps_counted_people_and_ignore_regions():
    text = "\n".join([
        "1,7,10,20,5,12,1,1,0,0",  # pedestrian, counted
        "1,8,30,40,6,6,1,2,0,1",  # people, counted
        "1,9,50,60,4,4,0,1,0,0",  # pedestrian, not counted
        "1,0,0,0,100,50,0,0,0,0",  # ignore region
        "2,7,11,21,5,12,1,1,0,0",
        "2,3,70,80,20,10,1,4,0,0",  # car
        "3,7,12,22,0,12,1,1,0,0",  # zero width
    ])  # fmt: skip
    frames = rg.parse_vid(text)
    people, ignore = frames[1]
    assert people.tolist() == [[10, 20, 5, 12, 7], [30, 40, 6, 6, 8]]
    assert ignore.tolist() == [[0, 0, 100, 50]]
    assert frames[2][0][:, 4].tolist() == [7] and len(frames[2][1]) == 0
    assert 3 not in frames


def test_sequences_pairs_frames_with_annotations():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in ("VisDrone2019-VID-val/sequences/s1/0000002.jpg",
                     "VisDrone2019-VID-val/sequences/s1/0000001.jpg",
                     "VisDrone2019-VID-val/annotations/s1.txt",
                     "VisDrone2019-VID-val/sequences/s2/0000001.jpg"):  # fmt: skip
            z.writestr(name, b"")
    with zipfile.ZipFile(buf) as z:
        seqs = rg.sequences(z)
    assert list(seqs) == ["s1"]  # s2 has no annotations
    assert [m.rsplit("/", 1)[-1] for m in seqs["s1"][1]] == [
        "0000001.jpg",
        "0000002.jpg",
    ]


def test_merge_drops_near_duplicates_from_overlapping_tiles():
    dets = np.array([[0, 0, 10, 10, 0.6], [1, 1, 10, 10, 0.9], [40, 40, 10, 10, 0.5]],
                    np.float32)  # fmt: skip
    out = rg.merge(dets)
    assert out[:, 4].tolist() == [np.float32(0.9), np.float32(0.5)]
