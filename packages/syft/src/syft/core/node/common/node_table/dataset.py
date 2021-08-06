# future
from __future__ import annotations

# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict as TypeDict
from typing import Optional

# third party
import numpy as np
import pandas as pd
from sqlalchemy import Column
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base
from .....lib.python.primitive_factory import PyPrimitive
from .....lib.python.string import String as SyftString
from ....common.serde.recursive import RecursiveSerde
from ....common.uid import UID
from ...lib.biblib.bib import Entry  # type: ignore
from ...lib.biblib.bib import Parser  # type: ignore


class Bibtex:
    def __init__(self, out: str) -> None:
        self.out = out

    def __str__(self) -> str:
        return self.out

    def _repr_html_(self) -> str:
        print(self.out)
        return f"<b>{self.out}</b>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Bibtex):
            return self.out == other.out
        return self == other


class DatasetDBObject(Base):
    __tablename__ = "dataset"

    # the unique id of the dataset used to identify this object by computers
    id = Column(String(256), primary_key=True)

    # the object store mapping string names -> pointer objects where the pointers
    # point to objects in the domain's object store (the actual private data of this dataset)
    private_assets = Column(JSON())

    # the (probably unique) recognizable string name of the dataset used to identify this object by people
    name = Column(String(256))

    # the free text description of this dataset
    description = Column(String(4096))

    citations = Column(JSON())

    bibtex = Column(String(4096))

    # any arbitrary public_metadata **kwargs added to this Dataset which we would like to include
    # in possible full-text string search becasue the key and value of the kwarg
    # is valid JSON (string -> string, string -> int, etc.)
    public_json_metadata = Column(JSON())

    # any arbitrary public_metadata **kwargs added to this Dataset which aren't valid json because
    # while the key might be a string, the value is some complex Python object which we have to
    # simply store as a blob and don't want to include in any full-text string search in the future
    # (even though the stored object is public and downloadable)
    public_blob_metadata = Column(JSON())


class Dataset(RecursiveSerde):

    json_representable = (str, int, bool, float)
    __attr_allowlist__ = [
        "id",
        "private_assets",
        "public_metadata",
        "name",
        "description",
        "metadata_names",
        "citations_dict",
    ]

    def __init__(
        self,
        *,
        private_assets: TypeDict[str, Any],
        name: str,
        description: str,
        bibtex: Optional[str] = None,
        id: Optional[UID] = None,
        **public_metadata: TypeDict[str, Any],
    ) -> None:
        if id is None:
            id = UID()
        self.id = id

        # this information is PRIVATE and MANDATORY (but can be empty dict)
        self.private_assets: TypeDict[str, Any] = private_assets

        # this information is PUBLIC and MANDATORY
        self.name = name
        self.description = description

        self.citations = {}
        if bibtex:
            self.citations = Parser().parse(str_or_fp_or_iter=bibtex).get_entries()
        self.metadata_names = ["name", "description"] + list(public_metadata.keys())

        for k, v in public_metadata.items():
            setattr(self, k, v)

    def __eq__(self, other: Any) -> bool:
        # values like Tensor or nd.array == return a Self not a bool
        def comp_dict(left: TypeDict, right: TypeDict) -> bool:
            # keys dont match so not equal
            if len(left.keys()) != len(right.keys()):
                return False
            for k in left.keys():
                # key is missing so not equal
                if k not in right.keys():
                    return False
                l_value = left[k]
                r_value = right[k]
                res = l_value == r_value
                # result is not a bool so it might be a Tensor or np.ndarray
                if not isinstance(res, bool):
                    if hasattr(res, "all"):
                        # check if all the values are True
                        res = res.all()
                    else:
                        # unknown type so cant compare
                        raise Exception(f"Unable to compare {k} result of == is: {res}")
                if res is False:
                    return res
            return True

        if isinstance(other, Dataset):
            return (
                self.id == other.id
                and comp_dict(self.private_assets, other.private_assets)
                and self.name == other.name
                and self.description == other.description
                and self.citations == other.citations
                and self.bibtex == other.bibtex
                and comp_dict(self._public_json_metadata, other._public_json_metadata)
                and comp_dict(self._public_blob_metadata, other._public_blob_metadata)
            )

        return self == other

    def to_db_object(self) -> DatasetDBObject:
        return DatasetDBObject(
            id=self.id,
            private_assets=self.private_assets,
            name=self.name,
            description=self.description,
            citations=self.citations_dict,
            bibtex=self.bibtex,
            public_json_metadata=self._public_json_metadata,
            public_blob_metadata=self._public_blob_metadata,
        )

    @staticmethod
    def from_db_object(dataset_db_obj: DatasetDBObject) -> Dataset:

        private_assets = dataset_db_obj.private_assets
        name = dataset_db_obj.name
        description = dataset_db_obj.description
        bibtex = dataset_db_obj.bibtex
        id = dataset_db_obj.id

        public_metadata = {}
        public_metadata.update(dataset_db_obj.public_json_metadata)
        public_metadata.update(dataset_db_obj.public_blob_metadata)
        public_metadata.pop("name")
        public_metadata.pop("description")

        return Dataset(
            private_assets=private_assets,
            name=name,
            description=description,
            bibtex=str(bibtex),
            id=id,
            **public_metadata,
        )

    @property
    def citations_dict(self) -> TypeDict:
        """I was lazy and didn't want to build a custom serializer, so I created this
        property and the setter below so that I could just use RecursiveSerde."""
        out_dict = {}

        for k, v in dict(self.citations).items():
            if isinstance(k, PyPrimitive):
                k = k.upcast()
            out_dict[k] = SyftString(v.to_bib())

        return out_dict

    @citations_dict.setter
    def citations_dict(self, new_dict: TypeDict) -> None:
        """I was lazy and didn't want to build a custom serializer, so I created this
        setter and the property above so that I could just use RecursiveSerde."""
        new_citations = {}
        for k, v in new_dict.items():
            if isinstance(v, PyPrimitive):
                v = v.upcast()
            new_citations[k] = list(
                Parser().parse(str_or_fp_or_iter=v).get_entries().values()
            )[0]

        self.citations = OrderedDict(new_citations)

    @property
    def _public_json_metadata(self) -> TypeDict:
        out = {}
        for key in self.metadata_names:
            if isinstance(key, PyPrimitive):
                key = key.upcast()

            val = getattr(self, key)
            if isinstance(val, PyPrimitive):
                val = val.upcast()

            if isinstance(val, Dataset.json_representable):
                out[key] = val

        return out

    @property
    def _public_blob_metadata(self) -> TypeDict:
        out = {}
        for key in self.metadata_names:
            if isinstance(key, PyPrimitive):
                key = key.upcast()

            val = getattr(self, key)
            if isinstance(val, PyPrimitive):
                val = val.upcast()

            if not isinstance(val, Dataset.json_representable):
                out[key] = val
        return out

    @property
    def public_metadata(self) -> TypeDict:
        """I was lazy and didn't want to build a custom serializer, so I created this property and
        the setter below so that I could just use RecursiveSerde"""
        out = {}
        for key in self.metadata_names:
            val = getattr(self, str(key))
            out[key] = val
        return out

    @public_metadata.setter
    def public_metadata(self, new_val: TypeDict) -> None:
        """I was lazy and didn't want to build a custom serializer, so I created this setter and
        the property above so that I could just use RecursiveSerde."""
        if not hasattr(self, "metadata_names"):
            self.metadata_names = ["name", "description"]
        self.metadata_names += list(str(key) for key in new_val.keys())
        for key, val in new_val.items():
            setattr(self, key, val)

    @property
    def pandas(self) -> pd.DataFrame:
        data = np.array(
            [str(self.private_assets.keys())] + list(self.public_metadata.values()),
            dtype=object,
        ).reshape(1, -1)
        columns = ["Private Assets"] + list(
            map(
                lambda x: x.replace("_", " ").capitalize(),
                list(self.public_metadata.keys()),
            )
        )
        return pd.DataFrame(data, columns=columns)

    def entry2citation_string(self, entity: Entry) -> str:

        pd.set_option("display.max_colwidth", 1000)
        out = ""

        for author in entity.authors():
            out += author.first + " " + author.last + ", "
        out = out[:-2] + ". "

        out += entity["title"].replace("\n", "") + ". "

        if "booktitle" in entity:
            out += "In <i>" + entity["booktitle"] + "</i> . "

        if "archiveprefix" in entity and "eprint" in entity:
            out += (
                "Available:"
                + str(entity["archiveprefix"])
                + ":"
                + str(entity["eprint"])
                + ". "
            )
            out += (
                "<a href='http://arxiv.org/abs/"
                + entity["eprint"]
                + "'>http://arxiv.org/abs/"
                + entity["eprint"]
                + "</a>"
            )

        if "url" in entity:
            out += "<a href='" + entity["url"] + "'>" + entity["url"] + "</a>"

        return out

    def citations2table(self, citations: TypeDict) -> pd.DataFrame:
        outs = list()
        for key in citations.keys():
            outs.append(self.entry2citation_string(citations[key]))

        data = np.array(outs, dtype=object)
        columns = ["Citations"]
        df = pd.DataFrame(data, columns=columns)
        df = df.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "left")]},
            ]
        )
        return df

    @property
    def refs(self) -> pd.DataFrame:
        return self.citations2table(self.citations)

    @property
    def bibtex(self) -> Bibtex:

        out = ""
        for cite in self.citations.values():
            out += cite.to_bib() + "\n\n"

        return Bibtex(out=out)

    def _repr_html_(self) -> str:
        out = "<h3>Dataset: " + self.name + "</h3>"
        out += self.description[0:1000] + "<br /><br /><br />"

        out += self.pandas._repr_html_()

        if len(self.citations) > 0:
            out += "<br />"
            out += self.refs._repr_html_()

        if hasattr(self, "sample") and isinstance(self.sample, pd.DataFrame):  # type: ignore
            out += "<br /><hr /><center><h4>Sample Data</h4></center><hr />"
            out += self.sample[0:3]._repr_html_()  # type: ignore

        if isinstance(out, PyPrimitive):
            out = out.upcast()

        return out
