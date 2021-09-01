import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class MorpheusBase:
    def __init__(
            self, lexicon_path: str, unimorph_path: str, lang: str, lang_short: str
    ):
        self.pos_tags = ["N", "ADJ", "V.PTCP", "V.CVB", "V.MSDR", "V"]
        self.lexicon = defaultdict(set)
        self.vocab = set()
        self.type = None

        lang_lex_path = Path(lexicon_path) / f"{lang}"
        lang_um_path = Path(unimorph_path) / f"{lang}"

        if lang_lex_path.is_file():
            self.type = "unimorph_inflect"
            with open(lang_lex_path, "r") as rf:
                for line in rf:
                    orig_token, orig_msd, perturb_token, _ = line.strip().split("\t")
                    self.lexicon[(orig_token, orig_msd)].add(perturb_token)
                    self.vocab.add(orig_token)

        elif lang_um_path.is_file():
            self.type = "unimorph_dict"
            form2lemma = {}
            lemma2form = defaultdict(set)
            with open(lang_um_path, "r") as rf:
                for line in rf:
                    if line.strip() == "":
                        continue
                    splits = line.strip().split("\t")
                    lemma, form, _ = splits
                    form2lemma[form] = lemma
                    lemma2form[lemma].add(form)
            for form, lemma in form2lemma.items():
                self.lexicon[(form, "")] = set(lemma2form[lemma]) - {form}
                self.vocab.add(form)

        logger.info(
            f"lang: {lang}, lang_short: {lang_short}, "
            f"vocab: {len(self.vocab)}, lexicon: {len(self.lexicon)}"
        )

    def get_inflections(
            self, orig_token: List[str], orig_lemma: List[str], orig_msd: List[str],
            constrain_pos: bool = True,
    ):
        # elements of form (i, inflections) where i is the token's position in the sequence
        token_inflections = []

        for i, (token, lemma, msd) in enumerate(zip(orig_token, orig_lemma, orig_msd)):
            pos_tag = None
            msd_labels = set(msd.split(";"))
            for tag in self.pos_tags:
                # V.PTCP is prefered over V
                if tag in msd_labels:
                    pos_tag = tag
                    break

            if self.type == "unimorph_dict":
                inflections = (i, list(self.lexicon.get((lemma, ""), set())))
            elif pos_tag and self.type == "unimorph_inflect":
                inflections = (i, list(self.lexicon.get((lemma, msd), set())))
            else:
                inflections = (i, [])
            inflections[1].append(token)

            random.shuffle(inflections[1])
            token_inflections.append(inflections)
            if not constrain_pos:
                logger.warning("non POS constrained inflection is not implemented")
                return []

        return token_inflections
