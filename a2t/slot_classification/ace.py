from collections import defaultdict
from pprint import pprint
from copy import deepcopy
import json
from tqdm import tqdm

from typing import List, Iterable, Dict, Union

import torch

from .data import SlotFeatures


class NotPreprocessedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ACEArgumentsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        filter_events: List[str] = None,
        create_negatives: bool = True,
        mark_trigger: bool = True,
        force_preprocess: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.instances: List[SlotFeatures] = []

        self.data_path = data_path
        self.create_negatives = create_negatives
        self.filter_events = filter_events
        self.mark_trigger = mark_trigger

        path_name = data_path.replace(".jsonl", "")
        path_name = f"{path_name}.prepro.{create_negatives}.jsonl"
        if self.mark_trigger:
            path_name = path_name.replace(".jsonl", ".trigger.jsonl")

        try:
            if force_preprocess:
                raise NotPreprocessedError("The dataset is first time loaded, or the preprocessing is forced.")
            self._from_preprocessed(path_name)
        except (NotPreprocessedError, Exception):
            self._load()
            self._save_preprocessed(path_name)

        if verbose:
            print(path_name)

        self.labels = list(set(inst.role for inst in self.instances))
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = self.labels.copy()

    def __getitem__(self, idx: int) -> SlotFeatures:
        return self.instances[idx]

    def __len__(self) -> int:
        return len(self.instances)

    def _save_preprocessed(self, path: str) -> None:
        with open(path, "wt") as f:
            for instance in self.instances:
                f.write(f"{json.dumps(instance.__dict__)}\n")

    def _from_preprocessed(self, path: str) -> None:
        with open(path, "rt") as f:
            self.instances = [SlotFeatures(**json.loads(line)) for line in f]

    def _load(self):
        with open(self.data_path, "rt") as data_f:
            for line in tqdm(data_f):
                instance = json.loads(line)
                entities = {entity["id"]: entity for entity in instance["entity_mentions"]}
                tokens = instance["tokens"]

                if not len(instance["event_mentions"]):
                    continue

                for event in instance["event_mentions"]:

                    if self.filter_events:
                        event_types = event["event_type"].split(":")
                        if all([event_types[0] not in self.filter_events, ":".join(event_types[:2]) not in self.filter_events]):
                            continue

                    sent_entities = {key: deepcopy(entity) for key, entity in entities.items()}

                    if self.mark_trigger:
                        sentence = " ".join(
                            tokens[: event["trigger"]["start"]]
                            + ["<trg>"]
                            + tokens[event["trigger"]["start"] : event["trigger"]["end"]]
                            + ["<trg>"]
                            + tokens[event["trigger"]["end"] :]
                        )
                    else:
                        sentence = " ".join(tokens)

                    for argument in event["arguments"]:

                        self.instances.append(
                            SlotFeatures(
                                docid=instance["sent_id"],
                                trigger=event["trigger"]["text"],
                                trigger_id=event["id"],
                                trigger_type=event["event_type"],
                                trigger_sent_idx=0,
                                arg=argument["text"],
                                arg_id=argument["entity_id"],
                                arg_type=entities[argument["entity_id"]]["entity_type"],
                                arg_sent_idx=0,
                                role=argument["role"],
                                pair_type=f"{event['event_type']}:{entities[argument['entity_id']]['entity_type']}",
                                context=sentence,
                            )
                        )

                        if argument["entity_id"] in sent_entities:
                            sent_entities.pop(argument["entity_id"])

                    if self.create_negatives:
                        for key, entity in sent_entities.items():

                            self.instances.append(
                                SlotFeatures(
                                    docid=instance["sent_id"],
                                    trigger=event["trigger"]["text"],
                                    trigger_id=event["id"],
                                    trigger_type=event["event_type"],
                                    trigger_sent_idx=0,
                                    arg=argument["text"],
                                    arg_id=key,
                                    arg_type=entity["entity_type"],
                                    arg_sent_idx=0,
                                    role="no_relation",
                                    pair_type=f"{event['event_type']}:{entity['entity_type']}",
                                    context=sentence,
                                )
                            )

    def to_dict(self, predictions: List[str]) -> Iterable[Dict[str, Union[str, int]]]:
        instances_copy = deepcopy(self.instances)
        inst_per_doc = defaultdict(list)
        for inst, pred in zip(instances_copy, predictions):
            inst.prediction = pred
            inst_per_doc[inst.docid].append(inst)

        with open(self.data_path, "rt") as f:
            for line in f:
                instance = json.loads(line)
                for event in instance["event_mentions"]:
                    event["arguments"] = []
                    for pred in inst_per_doc[instance["doc_id"]]:
                        if pred.trigger_id == event["id"] and pred.prediction not in [
                            "no_relation",
                            "OOR",
                        ]:
                            event["arguments"].append(
                                {
                                    "entity_id": pred.arg_id,
                                    "role": pred.prediction,
                                    "text": pred.arg,
                                }
                            )
                yield instance


if __name__ == "__main__":
    dataset = ACEArgumentsDataset(
        "data/ace/english/train.oneie.json", create_negatives=True, mark_trigger=True, force_preprocess=True
    )
    pprint(dataset[0])
