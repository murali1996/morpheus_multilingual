import argparse
import logging
import subprocess
import sys

import stanza
from iso639 import languages

sys.path.append('./resources')
import unimorph_inflect


def download_unimorph(lg: str):
    logging.info("downloading UniMorph dictionary for %s" % lg)
    subprocess.run("mkdir -p unimorph_dicts", shell=True)
    subprocess.run(
        "wget -q https://raw.githubusercontent.com/unimorph/%s/master/%s -O unimorph_dicts/%s" % (
            lg, lg, lg),
        shell=True,
    )


def download_unimorph_inflect_models(lg: str):
    logging.info("downloading unimorph_inflect models for %s" % lg)
    unimorph_inflect.download(lg)


def prepare_label_vocab(lg: str):
    logging.info("preparing label vocabulary for %s" % lg)
    subprocess.run("mkdir -p label_vocab", shell=True)
    subprocess.run(
        "python morpheus_multilingual/utils/build_label_vocab.py -d unimorph_dicts/%s "
        "-l morpheus_multilingual/utils/dimension2label.txt -v label_vocab/%s" % (lg, lg),
        shell=True,
    )


def download_stanza_model(two_letter_lg: str):
    logging.info("downloading stanza model for %s" % two_letter_lg)
    stanza.download(two_letter_lg)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    parser = argparse.ArgumentParser(description="setup language specific resources")
    parser.add_argument("langs", type=str, nargs="+", help="list of language codes (e.g. rus, deu)")

    args = parser.parse_args()

    for lg in args.langs:
        download_unimorph(lg)
        download_unimorph_inflect_models(lg)
        prepare_label_vocab(lg)

        two_letter_lg = languages.get(part3="deu").alpha2
        download_stanza_model(two_letter_lg)
