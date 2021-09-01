import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict


def load_um_vocab(file_path: Path):
    dim2label = defaultdict(list)
    label2dim = {}
    pos = set()
    with open(file_path, "r") as rf:
        for line in rf:
            splits = line.strip().split("\t")
            if splits[0] == "Part_of_Speech":
                pos.add(splits[2])
            else:
                dim2label[splits[0]].append(splits[-1])
                label2dim[splits[-1]] = splits[0]

    return dim2label, label2dim, pos


def get_label_vocab(dict_path: Path, schema_path: Path) -> Dict:
    vocab = {}
    missing_labels = set()

    # get UniMorph schema info
    dim2label, label2dim, pos_schema = load_um_vocab(schema_path)
    # load language dictionary
    lang_dict = defaultdict(list)
    with open(dict_path, "r") as rf:
        for line in rf:
            if line.strip() != "":
                type_, token_, msd_ = line.strip().split("\t")
                lang_dict[type_].append((token_, msd_))

    for type_ in lang_dict:
        for token_, msd_ in lang_dict[type_]:
            # pos, *labels = msd_.split(";")
            pos, labels = None, []
            for l_ in msd_.split(";"):
                if l_ in pos_schema:
                    if pos is None:
                        pos = l_
                    else:
                        if l_.startswith(pos):
                            pos = l_
                else:
                    labels.append(l_)

            if pos and pos not in vocab:
                vocab[pos] = defaultdict(set)
            for l_ in labels:
                try:
                    dim = label2dim[l_]
                    vocab[pos][dim].add(l_)
                except:
                    missing_labels.add(l_)

    logging.warning(f"{len(missing_labels)} unique labels are missing in the Schema")
    if len(missing_labels) > 0:
        logging.warning(f"{' '.join(list(missing_labels))}\n")

    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="build label vocabulary from UniMorph dictionaries"
    )
    parser.add_argument(
        "-d", type=Path, help="path to UniMorph dictionary of the language"
    )
    parser.add_argument(
        "-l", type=Path, help="path to all valid labels in UniMorph Schema"
    )
    parser.add_argument(
        "-v", type=Path, help="path to write label vocabulary of the language"
    )

    logging.basicConfig(format="%(levelname)s: %(message)s")
    args = parser.parse_args()

    vocab = get_label_vocab(args.d, args.l)

    with open(args.v, "w") as wf:
        for pos in vocab:
            for dim in vocab[pos]:
                wf.write(f"{pos}\t{dim}\t{' '.join(vocab[pos][dim])}\n")
