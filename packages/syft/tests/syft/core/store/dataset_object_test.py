# stdlib
from typing import Any
from typing import Dict as TypeDict

# third party
import torch as th

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.common import UID
from syft.core.node.common.node_table.dataset import Bibtex
from syft.core.node.common.node_table.dataset import Dataset
from syft.core.node.common.node_table.dataset import DatasetDBObject


def get_bibtex() -> str:
    return """
@inproceedings{DBLP:conf/ccs/AbadiCGMMT016,
  author    = {Martin Abadi and
               Andy Chu and
               Ian J. Goodfellow and
               H. Brendan McMahan and
               Ilya Mironov and
               Kunal Talwar and
               Li Zhang},
  editor    = {Edgar R. Weippl and
               Stefan Katzenbeisser and
               Christopher Kruegel and
               Andrew C. Myers and
               Shai Halevi},
  title     = {Deep Learning with Differential Privacy},
  booktitle = {Proceedings of the 2016 {ACM} {SIGSAC} Conference on Computer and
               Communications Security, Vienna, Austria, October 24-28, 2016},
  pages     = {308--318},
  publisher = {{ACM}},
  year      = {2016},
  url       = {https://doi.org/10.1145/2976749.2978318},
  doi       = {10.1145/2976749.2978318},
  timestamp = {Tue, 10 Nov 2020 20:00:49 +0100},
  biburl    = {https://dblp.org/rec/conf/ccs/AbadiCGMMT016.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


@misc{ryffel2018generic,
       title={A generic framework for privacy preserving deep learning},
       author={Theo Ryffel and Andrew Trask and Morten Dahl and Bobby Wagner and Jason
               Mancuso and Daniel Rueckert and Jonathan Passerat-Palmbach},
       year={2018},
       eprint={1811.04017},
       archivePrefix={arXiv},
       primaryClass={cs.LG}
}
"""


def private_assets() -> TypeDict[str, Any]:
    raw_data = th.Tensor([1, 2, 3])
    return {"feb2021": raw_data}


def create_dataset() -> Dataset:

    return Dataset(
        private_assets=private_assets(),
        name="iStat Trade Data - first 50K rows",
        description=(
            "A collection of reports from Italy's statistics bureau about how "
            + "much it thinks it imports and exports from other countries."
        ),
        bibtex=get_bibtex(),
        arbitary="things",
        public="places",
        metadata="stuff",
        some_num=99,
        sample_data=private_assets()["feb2021"][0:1],
    )


def test_dataset_db_object() -> None:
    dataset = create_dataset()

    db_obj = dataset.to_db_object()

    assert isinstance(db_obj, DatasetDBObject)
    assert isinstance(db_obj.id, UID)
    assert db_obj.id == dataset.id

    assert isinstance(db_obj.private_assets, dict)
    assert len(db_obj.private_assets) == 1
    assert (db_obj.private_assets["feb2021"] == private_assets()["feb2021"]).all()
    assert isinstance(db_obj.private_assets["feb2021"], th.Tensor)

    assert isinstance(db_obj.name, str)
    assert db_obj.name == dataset.name

    assert isinstance(db_obj.description, str)
    assert db_obj.description == dataset.description

    assert isinstance(db_obj.citations, dict)
    assert len(db_obj.citations.keys()) == 2
    assert db_obj.citations == dataset.citations_dict

    assert isinstance(db_obj.bibtex, Bibtex)
    assert db_obj.bibtex == dataset.bibtex

    assert isinstance(db_obj.public_json_metadata, dict)
    assert len(db_obj.public_json_metadata) == 6

    assert isinstance(db_obj.public_blob_metadata, dict)
    assert len(db_obj.public_blob_metadata) == 1

    dataset2 = Dataset.from_db_object(dataset_db_obj=db_obj)

    assert dataset == dataset2


def test_dataset_serde() -> None:
    dataset = create_dataset()
    proto = serialize(dataset)

    de = deserialize(proto)
    assert dataset == de

    blob = serialize(dataset, to_bytes=True)
    de2 = deserialize(blob, from_bytes=True)

    assert dataset == de2
