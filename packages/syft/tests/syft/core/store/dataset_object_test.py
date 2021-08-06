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

# raw_data = pd.read_csv('../../datasets/it - feb 2021.csv')[0:50000]


# dataset = Dataset(private_assets={"feb2021":raw_data},
#                     name="iStat Trade Data - first 50K rows",
#                     description="""A collection of reports from Italy's statistics
#                     bureau about how much it thinks it imports and exports from other countries.""",
#                     arbitary="aasdfads",
#                     public="asdfadsf",
#                     metadata="asdfasdfasdf",
#                     sample_data=raw_data[0:10]
#                     )

# you get back a pointer but GC is disabled (becasue you called load)
# istat_data = node.load_dataset(dataset)


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


# class Dataset(RecursiveSerde):

#     __attr_allowlist__ = [
#         "id",
#         "private_assets",
#         "public_metadata",
#         "name",
#         "description",
#         "metadata_names",
#         "citations_dict",
#     ]

#     def __init__(
#         self,
#         private_assets: Dict[str, Any],
#         name,
#         description,
#         bibtex: str,
#         id=None,
#         **public_metadata
#     ):

#         # Base.__init__(self)

#         if id is None:
#             id = UID()
#         self.id = id

#         # this information is PRIVATE and MANDATORY (but can be empty dict)
#         self.private_assets: Dict[str, Any] = private_assets

#         # this information is PUBLIC and MANDATORY
#         self.name = name
#         self.description = description

#         self.citations = Parser().parse(str_or_fp_or_iter=bibtex).get_entries()

#         self.metadata_names = ["name", "description"] + list(public_metadata.keys())

#         for k, v in public_metadata.items():
#             setattr(self, k, v)

#     def to_db_object(self):
#         return DatasetDBObject(
#             id=self.id,
#             private_assets=self.private_assets,
#             name=self.name,
#             description=self.description,
#             citations=self.citations_dict,
#             bibtex=self.bibtex,
#             public_json_metadata=self._public_json_metadata,
#             public_blob_metadata=self._public_blob_metadata,
#         )

#     @staticmethod
#     def from_db_object(dataset_db_obj: DatasetDBObject) -> Dataset:

#         private_assets = dataset_db_obj.private_assets
#         name = dataset_db_obj.name
#         description = dataset_db_obj.description
#         bibtex = dataset_db_obj.bibtex
#         id = dataset_db_obj.id

#         public_metadata = {}
#         public_metadata.update(dataset_db_obj.public_json_metadata)
#         public_metadata.update(dataset_db_obj.public_blob_metadata)

#     @property
#     def citations_dict(self):
#         """I was lazy and didn't want to build a custom serializer, so I created this property and
#         the setter below so that I could just use RecursiveSerde."""
#         out_dict = {}

#         for k, v in dict(self.citations).items():
#             # syft absolute
#             from syft.lib.python.string import String as SyftString

#             out_dict[k] = SyftString(v.to_bib())
#         return out_dict

#     @citations_dict.setter
#     def citations_dict(self, new_dict):
#         """I was lazy and didn't want to build a custom serializer, so I created this setter and
#         the property above so that I could just use RecursiveSerde."""
#         new_citations = {}
#         for k, v in new_dict.items():
#             if not isinstance(v, str):
#                 v = v.upcast()
#             new_citations[k] = list(
#                 Parser().parse(str_or_fp_or_iter=v).get_entries().values()
#             )[0]
#         self.citations = OrderedDict(new_citations)

#     @property
#     def _public_json_metadata(self):
#         out = {}
#         for key in self.metadata_names:
#             val = getattr(self, key)
#             if isinstance(val, (str, int, bool, float)):
#                 out[key] = val
#         return out

#     @property
#     def _public_blob_metadata(self):

#         out = {}
#         for key in self.metadata_names:
#             val = getattr(self, key)
#             if not isinstance(val, str):
#                 out[key] = val
#         return out

#     @property
#     def public_metadata(self):
#         """I was lazy and didn't want to build a custom serializer, so I created this property and
#         the setter below so that I could just use RecursiveSerde"""
#         out = {}
#         for key in self.metadata_names:
#             val = getattr(self, str(key))
#             out[key] = val
#         return out

#     @public_metadata.setter
#     def public_metadata(self, new_val):
#         """I was lazy and didn't want to build a custom serializer, so I created this setter and
#         the property above so that I could just use RecursiveSerde."""
#         for key, val in new_val.items():
#             setattr(self, key, val)

#     @property
#     def pandas(self):
#         data = np.array(
#             [str(self.private_assets.keys())] + list(self.public_metadata.values()),
#             dtype=object,
#         ).reshape(1, -1)
#         columns = ["Private Assets"] + list(
#             map(
#                 lambda x: x.replace("_", " ").capitalize(),
#                 list(self.public_metadata.keys()),
#             )
#         )
#         return pd.DataFrame(data, columns=columns)

#     def entry2citation_string(self, entity):

#         pd.set_option("display.max_colwidth", 1000)
#         out = ""

#         for author in entity.authors():
#             out += author.first + " " + author.last + ", "
#         out = out[:-2] + ". "

#         out += entity["title"].replace("\n", "") + ". "

#         if "booktitle" in entity:
#             out += "In <i>" + entity["booktitle"] + "</i> . "

#         if "archiveprefix" in entity and "eprint" in entity:
#             out += (
#                 "Available:"
#                 + str(entity["archiveprefix"])
#                 + ":"
#                 + str(entity["eprint"])
#                 + ". "
#             )
#             out += (
#                 "<a href='http://arxiv.org/abs/"
#                 + entity["eprint"]
#                 + "'>http://arxiv.org/abs/"
#                 + entity["eprint"]
#                 + "</a>"
#             )

#         if "url" in entity:
#             out += "<a href='" + entity["url"] + "'>" + entity["url"] + "</a>"

#         return out

#     def citations2table(self, citations):
#         outs = list()
#         for key in citations.keys():
#             outs.append(self.entry2citation_string(citations[key]))

#         data = np.array(outs, dtype=object)
#         columns = ["Citations"]
#         df = pd.DataFrame(data, columns=columns)
#         df = df.style.set_table_styles(
#             [
#                 {"selector": "th", "props": [("text-align", "center")]},
#                 {"selector": "td", "props": [("text-align", "left")]},
#             ]
#         )
#         return df

#     @property
#     def refs(self):
#         return self.citations2table(self.citations)

#     @property
#     def bibtex(self):

#         out = ""
#         for cite in self.citations.values():
#             out += cite.to_bib() + "\n\n"

#         class Bibtex:
#             def __str__(self):
#                 return out

#             def _repr_html_(self):
#                 print(out)
#                 return

#         return Bibtex()

#     def _repr_html_(self):
#         out = "<h3>Dataset: " + self.name + "</h3>"
#         out += self.description[0:1000] + "<br /><br /><br />"

#         out += self.pandas._repr_html_()

#         if len(self.citations) > 0:
#             out += "<br />"
#             out += self.refs._repr_html_()

#         if hasattr(self, "sample") and isinstance(self.sample, pd.DataFrame):
#             out += "<br /><hr /><center><h4>Sample Data</h4></center><hr />"
#             out += self.sample[0:3]._repr_html_()

#         if not isinstance(out, str):
#             out = out.upcast()

#         return out
