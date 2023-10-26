
import argparse
from typing import Any
import ast
import json
import cattrs
from nltk import ngrams
from collections import defaultdict
from typing import List, Tuple
from dataclasses import dataclass
from typing import List, Optional, Dict

PART_INPUT: str = "input"
PART_REF: str = "references"

@dataclass(frozen=True)
class ObjectSpec:
    """Specifies how to construct an object."""

    # Class name of an object
    class_name: str

    # Arguments used to construct the scenario
    args: Dict[str, Any]

    def __hash__(self):
        return hash((self.class_name, tuple((k, self.args[k]) for k in sorted(self.args.keys()))))

class ScenarioSpec(ObjectSpec):
    pass

@dataclass(frozen=True)
class ScenarioSpecInstanceIds:
    """
    Instance ids associated with a scenario
    """

    scenario_spec: ScenarioSpec

    instance_ids: List[str]


@dataclass(frozen=True)
class LightScenarioKey:
    """
    Key for LightScenario
    """

    scenario_spec: ScenarioSpec

    split: str

    def __hash__(self):
        return hash((self.scenario_spec, self.split))


@dataclass(frozen=True)
class OverlapProtocolSpec:
    """Specification for how we compute overlap"""

    # the N of the n_grams we're running
    n: int


@dataclass(frozen=True)
class DataOverlapStatsKey:
    """Dataclass that represents output data overlap stats"""

    light_scenario_key: LightScenarioKey

    overlap_protocol_spec: OverlapProtocolSpec


@dataclass(frozen=True)
class DataOverlapStats:
    """Dataclass that represents output data overlap stats"""

    data_overlap_stats_key: DataOverlapStatsKey

    num_instances: int

    instance_ids_with_overlapping_input: List[str]

    instance_ids_with_overlapping_reference: List[str]


@dataclass(frozen=True)
class EntryDataOverlapKey:
    """Unique key representing either the input or references of a single instance in a scenario."""

    stats_key: DataOverlapStatsKey
    part: str
    """Either PART_INPUT or PART_REF"""
    instance_id: str


@dataclass(frozen=True)
class EntryOverlapNgrams:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlapping_ngram_counts: List[Tuple[str, int]]
    
@dataclass(frozen=True)
class LightInstance:
    """
    A lighter `Instance` with only text fields.
    """

    input: str
    """The input"""

    references: List[str]
    """References that help us evaluate"""

    id: Optional[str] = None
    """Helm instance id"""

    
@dataclass(frozen=True)
class LightScenario:
    """
    A lighter `Scenario`.
    """

    scenario_key: LightScenarioKey

    instances: List[LightInstance]
    """Instances of this scenario"""


def load_light_scenarios_from_jsonl(path: str) -> List[LightScenario]:
    """
    Create a list of light scenarios from a jsonl file, where each json represents a LightScenario object.

    Input file format:

    Instance JSON 1
    Instance JSON 2
    Instance JSON 3
    ...
    """

    def create_light_instance_from_dict(instance_dict: dict) -> LightInstance:
        return LightInstance(
            input=instance_dict[PART_INPUT], references=instance_dict[PART_REF], id=instance_dict["id"]
        )

    light_scenarios: List[LightScenario] = []
    light_scenario_jsons = open(path, "r").readlines()
    for light_scenario_json in light_scenario_jsons:
        light_scenario_dict: dict = json.loads(light_scenario_json)

        light_scenario_key_dict: dict = light_scenario_dict["scenario_key"]
        # if the light_scenarios are exported from helm, they will have a scenario_spec field
        scenario_spec = ScenarioSpec(**light_scenario_key_dict["scenario_spec"])
        light_scenario_key = LightScenarioKey(scenario_spec=scenario_spec, split=light_scenario_key_dict["split"])
        light_instances: List[LightInstance] = [
            create_light_instance_from_dict(instance_dict) for instance_dict in light_scenario_dict["instances"]
        ]
        light_scenarios.append(LightScenario(scenario_key=light_scenario_key, instances=light_instances))
    return light_scenarios

from typing import Any, Dict

from dataclasses import asdict, is_dataclass

def asdict_without_nones(obj: Any) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})



from enum import Enum
import re

from typing import List
from string import punctuation


class LightTokenizer:
    """
    Tokenize texts by splitting on whitespaces.
    """

    def tokenize(self, text: str) -> List[str]:
        return text.split()


class DefaultTokenizer(LightTokenizer):
    """
    Normalize and tokenize texts by converting all characters to the lower case and
    splitting on whitespaces and punctuations.
    """

    def __init__(self):
        super().__init__()
        self.r = re.compile(r"[\s{}]+".format(re.escape(punctuation)))

    def tokenize(self, text: str) -> List[str]:
        return self.r.split(text.lower())


def get_tokenizer(normalization) -> LightTokenizer:
    if normalization == "none":
        return LightTokenizer()
    elif normalization == "default":
        return DefaultTokenizer()
    else:
        raise ValueError(f"Normalization strategy {normalization} is not defined.")



@dataclass(frozen=True)
class EntryDataOverlapKey:
    """Unique key representing either the input or references of a single instance in a scenario."""

    stats_key: DataOverlapStatsKey
    part: str
    """Either PART_INPUT or PART_REF"""
    instance_id: str


# Input: List[EntryOverlapNgrams]
@dataclass(frozen=True)
class EntryOverlapNgrams:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlapping_ngram_counts: List[Tuple[str, int]]


class PartialOverlapSpec(int, Enum):
    binary = 0
    jaccard = 1
    token = 2

@dataclass(frozen=True)
class FrequencySpec:
    # Filter ngrams with frequency >= filter_value; 0 means no filter
    filter_value: int
    # Whether to apply weight; we'll do inverse frequency
    weighting: bool
        
@dataclass(frozen=True)
class MetricProtocolSpec:
    """Specification for how we compute the metric"""
    
    partial_overlap_spec: PartialOverlapSpec
    frequency_spec: FrequencySpec
        
@dataclass(frozen=True)
class OverlapMetric:
    metric_score: float # use 0/1 for binary, can revise as neded
    metric_protocol_spec: MetricProtocolSpec

# Output: List[EntryOverlapMetric]
@dataclass(frozen=True)
class EntryOverlapMetric:
    """Dataclass that represents output data overlap stats"""

    entry_data_overlap_key: EntryDataOverlapKey

    overlap_metric: OverlapMetric

def get_metrics(ngrams_path, scenario_path, out_path, filter_path, N):

    scenario_spec_instance_id_dict = dict()
    if filter_path:
        scenario_spec_instance_ids_json = filter_path
        scenario_spec_instance_ids_jsons = open(scenario_spec_instance_ids_json, "r").readlines()
        for scenario_spec_instance_ids_json in scenario_spec_instance_ids_jsons:
            scenario_spec_instance_ids_dict = json.loads(scenario_spec_instance_ids_json)
            scenario_spec_instance_ids = cattrs.structure(scenario_spec_instance_ids_dict, ScenarioSpecInstanceIds)
            scenario_spec_instance_id_dict[
                scenario_spec_instance_ids.scenario_spec
            ] = scenario_spec_instance_ids.instance_ids

    # Read Ngrams
    ngram_jsons = open(ngrams_path, "r").readlines()
    entry_overlap_ngrams_list = []
    for ngram_json in ngram_jsons:
        entry_overlap_ngrams = json.loads(ngram_json)
        entry_overlap_ngrams = cattrs.structure(entry_overlap_ngrams, EntryOverlapNgrams)
        scenario_spec = entry_overlap_ngrams.entry_data_overlap_key.stats_key.light_scenario_key.scenario_spec
        if scenario_spec_instance_id_dict:
            if scenario_spec not in scenario_spec_instance_id_dict:
                continue
            instance_ids = scenario_spec_instance_id_dict[scenario_spec]
            if entry_overlap_ngrams.entry_data_overlap_key.instance_id not in instance_ids:
                continue
            else:
                entry_overlap_ngrams_list.append(entry_overlap_ngrams)
        else:
                entry_overlap_ngrams_list.append(entry_overlap_ngrams)




    def merge_entries(entry_overlap_ngrams_list):
        overlapping_counts = defaultdict(int)
        for entry_overlap_ngrams in entry_overlap_ngrams_list:
            entry_data_overlap_key = entry_overlap_ngrams.entry_data_overlap_key
            overlapping_ngram_counts = entry_overlap_ngrams.overlapping_ngram_counts
            for ngram, count in overlapping_ngram_counts:
                overlapping_counts[ngram] += count
        overlapping_ngram_counts_list = []
        for ngram, count in overlapping_counts.items():
            overlapping_ngram_counts_list.append((ngram, count))
        return [EntryOverlapNgrams(
                        entry_data_overlap_key=entry_data_overlap_key, overlapping_ngram_counts=overlapping_ngram_counts_list
                    )]

    # create entry_overlap_ngrams_dict, a dict of entry_data_overlap_key -> EntryOverlapNgrams
    entry_overlap_ngrams_dict = defaultdict(list)
    for entry_overlap_ngrams in entry_overlap_ngrams_list:
        entry_data_overlap_key = entry_overlap_ngrams.entry_data_overlap_key
        overlapping_ngram_counts = entry_overlap_ngrams.overlapping_ngram_counts
        ngram_count = entry_data_overlap_key.stats_key.overlap_protocol_spec.n
        if ngram_count not in [N]:
            continue
        entry_overlap_ngrams_dict[entry_data_overlap_key].append(entry_overlap_ngrams)
        
        # We need to merge entries if sharded by training data, since there'll be redundancy
        # Can refactor to no list later
        if len(entry_overlap_ngrams_dict[entry_data_overlap_key]) > 1:
            entry_overlap_ngrams_dict[entry_data_overlap_key] = merge_entries(entry_overlap_ngrams_dict[entry_data_overlap_key])

    # Read Scenarios
    light_scenarios = load_light_scenarios_from_jsonl(scenario_path)
    light_scenario_instance_dict = dict()
    for light_scenario in light_scenarios:
        instances = light_scenario.instances
        instance_dict = dict()
        for instance in instances:
            instance_dict[instance.id] = instance
        light_scenario_instance_dict[light_scenario.scenario_key] = instance_dict


    def compute_binary_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute  binary overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        metric_score = 0

        for ngram in ngrams(tokens, N):
            count = ngram_counts_dict[ngram]
            if frequency == 0 or count <= frequency:
                if count != 0:
                    metric_score = 1
                    break

        overlap_metric = OverlapMetric(
            metric_score = metric_score,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.binary,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        return overlap_metric

    def compute_jaccard_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute weighted and unweighted jaccard overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        total_ngram_count = 0
        counts = 0
        weighted_score = 0

        for ngram in ngrams(tokens, N):
            count = ngram_counts_dict[ngram]
            if frequency == 0 or count <= frequency:
                if count != 0:
                    counts += 1
                    weighted_score += 1 / count
            total_ngram_count += 1

        unweighted_score = counts / total_ngram_count
        weighted_score = weighted_score / total_ngram_count

        unweighted_overlap_metric = OverlapMetric(
            metric_score = unweighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.jaccard,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        weighted_overlap_metric = OverlapMetric(
            metric_score = weighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.jaccard,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = True
                )
            )
        )

        return unweighted_overlap_metric, weighted_overlap_metric

    # Token overlap
    def compute_token_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency = 0):
        """ 
        Compute weighted and unweighted token overlap
        If pass in frequency, include only the ngrams with count <= frequency
        """
        tokens = tokenizer.tokenize(instance_str)
        ngram_counts_dict = defaultdict(int)
        
        # construct a dict of ngram -> count
        for ngram, count in overlapping_ngram_counts:
            ngram = tuple(ast.literal_eval(ngram))
            ngram_counts_dict[ngram] = count

        total_token_count = 0
        counts = 0
        weighted_score = 0
        weight = 0
        token_budget = 0

        for ngram in ngrams(tokens, N):
            curr_count = ngram_counts_dict[ngram]

            # either no frequency, or check current count is less than frequency
            # or a previous contiguous count (weight != 0) less than frequency
            if frequency == 0 or curr_count <= frequency or (weight != 0 and weight <= frequency):
                if curr_count != 0:
                    token_budget = N
                    if weight > 0:
                        weight = min(curr_count, weight)
                    else:
                        weight = curr_count 

            if token_budget > 0:
                token_budget -= 1
                counts += 1
                weighted_score += 1 / weight
            else:
                weight = 0
            total_token_count += 1

        for token in ngram[1:]:
            if token_budget > 0:
                token_budget -= 1
                counts += 1
                weighted_score += 1 / weight
            total_token_count += 1

        unweighted_score = counts / total_token_count
        weighted_score = weighted_score / total_token_count

        unweighted_overlap_metric = OverlapMetric(
            metric_score = unweighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.token,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = False
                )
            )
        )

        weighted_overlap_metric = OverlapMetric(
            metric_score = weighted_score ,
            metric_protocol_spec = MetricProtocolSpec(
                partial_overlap_spec = PartialOverlapSpec.token,
                frequency_spec = FrequencySpec(
                    filter_value = frequency,
                    weighting = True
                )
            )
        )

        return unweighted_overlap_metric, weighted_overlap_metric

    def compute_and_add_metrics(instance_str, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency = 0):

        overlap_metric = compute_binary_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        binary_metric = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=overlap_metric)
        entry_overlap_metric_list.append(binary_metric)

        unweighted_overlap_metric, weighted_overlap_metric = compute_jaccard_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        unweighted_jaccard = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=unweighted_overlap_metric)
        weighted_jaccard = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=weighted_overlap_metric)
        entry_overlap_metric_list.append(unweighted_jaccard)
        entry_overlap_metric_list.append(weighted_jaccard)

        unweighted_overlap_metric, weighted_overlap_metric = compute_token_overlap(instance_str, overlapping_ngram_counts, tokenizer, frequency)
        unweighted_token = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=unweighted_overlap_metric)
        weighted_token = EntryOverlapMetric(entry_data_overlap_key=entry_data_overlap_key, overlap_metric=weighted_overlap_metric)
        entry_overlap_metric_list.append(unweighted_token)
        entry_overlap_metric_list.append(weighted_token)

    def save_metrics_to_jsonl(overlap_metrics: List[EntryOverlapMetric], filename: str):
        with open(filename, "w") as f:
            for overlap_metric in overlap_metrics:
                f.write(json.dumps(asdict_without_nones(overlap_metric), ensure_ascii=False) + "\n")


    entry_overlap_metric_list = []
    tokenizer = get_tokenizer('default')
    for entry_data_overlap_key, entry_overlap_ngrams_list in entry_overlap_ngrams_dict.items():
        data_overlap_stats_key = entry_data_overlap_key.stats_key
        light_scenario_key = data_overlap_stats_key.light_scenario_key
        instance_dict = light_scenario_instance_dict[light_scenario_key]
        for entry_overlap_ngrams in entry_overlap_ngrams_list:
            entry_data_overlap_key = entry_overlap_ngrams.entry_data_overlap_key
            instance_id = entry_data_overlap_key.instance_id
            instance = instance_dict[instance_id]
            part = entry_data_overlap_key.part
            overlapping_ngram_counts = entry_overlap_ngrams.overlapping_ngram_counts
            if part == 'input':
                compute_and_add_metrics(instance.input, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list)
                compute_and_add_metrics(instance.input, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency=10)
            if part == 'references':
                reference = ' '.join(instance.references)
                compute_and_add_metrics(reference, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list)
                compute_and_add_metrics(reference, overlapping_ngram_counts, tokenizer, entry_data_overlap_key, entry_overlap_metric_list, frequency=10)

    save_metrics_to_jsonl(entry_overlap_metric_list, out_path)
    return entry_overlap_metric_list



def get_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrams-path", type=str, required=True, help="Path to your ngrams")
    parser.add_argument("--scenario-path", type=str, required=True, help="Path to scenario data (benchmarking data)")
    parser.add_argument("--out-path", type=str, required=True, help="Path to the output metrics file")
    parser.add_argument("--filter-path", type=str, default='', help="Path to file for filtering a subset of tests")
    parser.add_argument("--N", type=int, default=13, help="N of input ngrams")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Call the get_metrics function with the constructed arguments
    get_metrics(args.ngrams_path, args.scenario_path, args.out_path, args.filter_path, args.N)

