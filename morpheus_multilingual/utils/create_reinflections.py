import argparse
import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

sys.path.append("../../resources")
sys.path.append("./resources")
from unimorph_inflect import inflect

logger = logging.getLogger(__name__)


def _get_reinflections(
        token: str, msd: str, label2dim: Dict, dim2label: Dict, all_pos: set, lg: str
) -> List:
    logging.debug(f"finding perturbations for token: {token}, msd: {msd}")
    candidates = []

    pos, flex_labels, fixed_labels = None, [], []
    for l_ in msd.split(";"):
        if l_ in label2dim:
            flex_labels.append(l_)
        elif l_ in all_pos:
            pos = l_
        else:
            fixed_labels.append(l_)

    if pos is None:
        logging.debug(f"no UniMorph POS found in {msd}")
        return []

    flex_pertb_labels = []
    for orig_l in flex_labels:
        dim = label2dim.get(orig_l, None)
        if dim:
            dim_labels = dim2label.get((pos, dim), set())
            if len(dim_labels) > 0:
                flex_pertb_labels.append(list(dim_labels))

    new_msd_tags = []
    for pertb_labels in itertools.product(*flex_pertb_labels):
        # new_labels = fixed_labels + list(pertb_labels)
        new_labels = list(pertb_labels)
        if len(new_labels) > 0:
            new_msd = f"{pos};{';'.join(new_labels)}"
            new_msd_tags.append(new_msd)

    logging.debug(f"# max new tags: {len(new_msd_tags)}")
    new_tokens = inflect([token] * len(new_msd_tags), new_msd_tags, language=lg)
    for new_token, new_msd in zip(new_tokens, new_msd_tags):
        candidates.append((token, msd, new_token, new_msd))

    return candidates


def get_reinflections(lang, input_file_path, output_file_path, label_vocab_path):
    label2dim, dim2label = {}, {}
    all_pos = set()
    with open(label_vocab_path, "r") as rf:
        for line in rf:
            pos, dim, labels = line.strip().split("\t")
            all_pos.add(pos)
            labels = labels.split()
            dim2label[(pos, dim)] = set(labels)
            for l_ in labels:
                if l_ in label2dim:
                    assert (label2dim[l_] == dim), \
                        f"mismatch in label2dim, {dim} vs {label2dim[l_]}"
                label2dim[l_] = dim

    with open(input_file_path, "r") as rf, open(output_file_path, "w") as wf:
        for idx, line in tqdm(enumerate(rf), disable=logger.level > 20):
            token, msd = line.strip().split()
            pertb_tokens = _get_reinflections(token, msd, label2dim, dim2label, all_pos, lang)
            for o_t, o_msd, p_t, p_msd in pertb_tokens:
                if o_t == p_t:
                    continue
                wf.write(f"{o_t}\t{o_msd}\t{p_t}\t{p_msd}\n")

    logger.info(f"Candidates' reinflections saved at: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument(
        "-l", "--lang", type=str,
        help="language code")
    parser.add_argument(
        "-f", "--input_file_path", type=Path,
        help="input path to source tokens with MSD")
    parser.add_argument(
        "-o", "--output_file_path", type=Path, required=False, default=None,
        help="path to write output (perturbed) tokens")
    parser.add_argument(
        "-v", "--label_vocab_path", type=Path, required=False, default=None,
        help="path to input language specific vocabulary"
    )

    args = parser.parse_args()

    logger = logging.getLogger(__file__)
    if logger.level == 0:
        logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    if not args.label_vocab_path:
        args.label_vocab_path = f"./label_vocab/{args.lang}"
    if not args.output_file_path:
        dir_, name_ = os.path.split(args.input_file_path)
        args.output_file_path = os.path.join(dir_, name_ + ".reinflected")

    logger.info(f"Finding reinflections for candidates obtained from stanza processing ...")
    get_reinflections(lang=args.lang, input_file_path=args.input_file_path,
                      output_file_path=args.output_file_path,
                      label_vocab_path=args.label_vocab_path)
    logger.info(f"Successfully computed all candidate reinflections")
