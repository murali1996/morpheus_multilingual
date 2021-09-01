import argparse
import logging
import os
from pathlib import Path

import torch
from iso639 import languages
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm

from morpheus_multilingual import (
    get_candidates,  # for candidates generation
    get_reinflections,  # for candidates' reinflections generation
    get_sentences_meta,  # utils to load stanza married file
    MorpheusFairseqTransformerNMT  # adversarial class for NMT task
)

logger = logging.getLogger(__name__)


def process_args(args):
    if not args.label_vocab_path:
        args.label_vocab_path = f"./label_vocab/{args.lang}"

    for name in dir(args):
        if getattr(args, name) and (
                not name.startswith("__") and
                (name.endswith("folder") or name.endswith("path"))
        ):
            assert os.path.exists(getattr(args, name)), logger.error(
                f"specified path/folder unavailable: name- {name}, value- {getattr(args, name)}"
            )

    if args.reinflection_lexicons_file_path and args.label_vocab_path:
        raise ValueError(
            f"Only one of reinflection_lexicons_file_path ({args.reinflection_lexicons_file_path})"
            f" and label_vocab_path ({args.label_vocab_path}) can be inputted at a time")

    if not args.use_pretrained_fairseq_model:

        for ff in [
            "checkpoint_best.pt",
            f"dict.{args.lang}.txt",
            f"dict.eng.txt",
            "spm8000.model",
        ]:
            assert os.path.exists(os.path.join(args.model_checkpoint_folder, ff)), logger.error(
                f"{ff} file missing in the checkpoints folder {args.model_checkpoint_folder}. "
                f"Available files are: {os.listdir(args.model_checkpoint_folder)}"
            )


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group(
        "create_adversaries",
        "The source language and the XXX-ENG TED formatted file path for finding adversaries"
    )
    group1.add_argument(
        "--lang", "-l", type=str, required=True,
        help="3-letter language code (ISO 639-2 Code)")
    group1.add_argument(
        "--mt_file_path", "-f", type=str, required=True,
        help="TED-corpus formatted file used for NMT")

    group2 = parser.add_argument_group(
        "create_dictionary", "Inputs required for candidate generation"
    )
    group2.add_argument(
        "--ud_tool_folder", "-ud", default="./resources/ud-compatibility/UD_UM", type=str,
        required=False,
        help="path where the UD_UM sub-directory of ud-compatibility tool's directory is located")
    group2.add_argument(
        "--unimorph_dicts_folder", "-uni", default="./unimorph_dicts", type=str, required=False,
        help="path where the unimorph dictionaries are located"
    )
    group2.add_argument(
        "--stanza_married_file_path", "-s", default="", type=str, required=False,
        help="path where the stanza married file corresponding to inputted mt_file was dumped "
             "from a previously run candidate generation step "
    )
    group2.add_argument(
        "--candidates_file_path", "-c", default="", type=str, required=False,
        help="path where the candidates obtained after stanza processing are dumped"
    )

    group3 = parser.add_argument_group(
        "reinflections", "Inputs required for obtaining candidates' reinflections"
    )
    group3.add_argument(
        "--reinflection_lexicons_file_path", "-r", default="", type=str,
        required=False,
        help="File path where the reinflections are dumped")
    group3.add_argument(
        "--label_vocab_path", "-v", default=None, type=Path, required=False,
        help="path to input language specific vocabulary")

    group4 = parser.add_argument_group(
        "create_adversaries", "Inputs for instantiating a NMT model to run inferences"
    )
    group4.add_argument(
        "--model_checkpoint_folder", "-m", type=str, required=True,
        help="A folder path consisting of files from a fairseq (pre)trained model")
    group4.add_argument(
        "--use_pretrained_fairseq_model", action="store_true",
        help="if True, the MorpheusFairseqTransformerNMTRand loads a pretrained fairseq model; "
             "available only for German (deu) and Russian (rus)",
    )

    group5 = parser.add_argument_group(
        "outputs", "Paths to save outputs"
    )
    group5.add_argument(
        "--outputs_folder", "-o", default="", type=str, required=False,
        help="path where output files are to be dumped"
    )
    group5.add_argument(
        "--logs_folder", default="", type=str, required=False,
        help="folder path to write log file",
    )

    groupN = parser.add_argument_group("extras", "extras")
    groupN.add_argument(
        "--use_chrf", action="store_true",
        help="use chrf scores instead of the default bleu",
    )
    groupN.add_argument(
        "--batch_size", "-b", default=8, type=int, required=True,
        help="batch size to do translations using fairseq",
    )
    groupN.add_argument(
        "--max_count", default=50, type=int, required=False,
        help="maximum number of times to query for a perturbed sentence in morpheus",
    )

    parser.set_defaults(func=process_args)

    args = parser.parse_args()
    args.func(args)

    candidates_files_output_folder, src_file_name = os.path.split(args.mt_file_path)

    """ Stanza processed data loading """

    logger.info(f"Obtaining Stanza processed outputs on the inputted mt_file: {args.mt_file_path}")
    if not (args.stanza_married_file_path and args.candidates_file_path):
        logger.info(f"Pretokenizing inputs using MosesTokenizer beofre passing to Stanza pipeline")
        tokenizer = MosesTokenizer(lang=languages.get(part3=args.lang).alpha2)
        all_untokenized_lines = [line for line in open(args.mt_file_path)]
        all_lines = [line for line in open(args.mt_file_path)]
        targets = [line.split("|||")[1].strip() for line in all_lines]
        sources = [line.split("|||")[0].strip() for line in all_lines]
        sources = [" ".join(tokenizer.tokenize(src)) for src in sources]
        all_lines = [f"{x} ||| {y}" for x, y in zip(sources, targets)]
        logger.info(f"Beginning stanza pipeline processing ...")
        (
            stanza_save_file_path,
            stanza_married_save_file_path,
            stanza_dict_file_path
        ) = get_candidates(
            args.lang,
            args.ud_tool_folder,
            args.unimorph_dicts_folder,
            candidates_files_output_folder,
            tokenize_pretokenized=True,
            all_lines=all_lines,
            all_untokenized_lines=all_untokenized_lines,
            src_file_name=src_file_name,
            use_gpu=True if "cuda" in DEVICE.lower() else False,
            save_dicts=True,
        )
        logger.info(f"Stanza outputs saved at {candidates_files_output_folder}")
        if not args.stanza_married_file_path:
            args.stanza_married_file_path = stanza_married_save_file_path
        if not args.candidates_file_path:
            args.candidates_file_path = stanza_dict_file_path

    sentences_meta = get_sentences_meta(args.stanza_married_file_path)
    logger.info(f"Stanza processed file loaded successfully from {args.stanza_married_file_path}")

    """ reformat metadata to find adversaries """

    if not args.outputs_folder:
        logger.info(f"No outputs folder specified; using the mt_file's folder to dump outputs")
        args.outputs_folder = os.path.join(candidates_files_output_folder, "adversaries")
    logger.info(f"Adversaries are being dumped in the folder: {args.outputs_folder}")
    os.makedirs(args.outputs_folder, exist_ok=True)

    if not args.logs_folder:
        logger.info(f"No outputs folder specified; using the mt_file's folder to dump outputs")
        args.logs_folder = os.path.join(candidates_files_output_folder, "adversaries_logs")
    logger.info(f"Adversaries are being dumped in the folder: {args.logs_folder}")
    os.makedirs(args.logs_folder, exist_ok=True)

    n_new = 0
    tokenizer = MosesTokenizer(lang=languages.get(part3=args.lang).alpha2)
    detokenizer = MosesDetokenizer(lang=languages.get(part3=args.lang).alpha2)
    source_sents = []
    target_sents = []
    source_sents_tokenized = []
    source_sents_lemma = []
    source_sents_msd = []
    for im, meta in tqdm(enumerate(sentences_meta), total=len(sentences_meta),
                         disable=logger.level != 0):
        meta_parts = meta.split("\n")
        sid, stext, strans, slines = (
            meta_parts[0],
            meta_parts[1],
            meta_parts[2],
            meta_parts[3:],
        )
        assert stext.startswith("# text = ")
        assert strans.startswith("# translation = ")
        stext, strans = stext[9:], strans[16:]
        stext_tokenized, stext_lemmas, stext_MSDs = [], [], []
        for sline in slines:
            sparts = sline.split("\t")
            org_word, lemma, msd = sparts[1], sparts[2], sparts[5]
            stext_tokenized.append(org_word)
            stext_lemmas.append(lemma)
            stext_MSDs.append(msd)
        source_sents.append(stext)
        target_sents.append(strans)
        source_sents_tokenized.append(stext_tokenized)
        source_sents_lemma.append(stext_lemmas)
        source_sents_msd.append(stext_MSDs)

    """ create reinflections if do not exist """

    if not args.reinflection_lexicons_file_path:
        logger.info(f"Finding reinflections for candidates obtained from stanza processing ...")
        dir_, name_ = os.path.split(args.mt_file_path)
        args.reinflection_lexicons_file_path = os.path.join(dir_, name_ + ".reinflected")
        get_reinflections(lang=args.lang, input_file_path=args.candidates_file_path,
                          output_file_path=args.reinflection_lexicons_file_path,
                          label_vocab_path=args.label_vocab_path)
        logger.info(f"Successfully computed all candidate reinflections")

    """ load model """

    morpheusNMT = MorpheusFairseqTransformerNMT(
        args.model_checkpoint_folder,
        args.lang,
        languages.get(part3=args.lang).alpha2,
        args.reinflection_lexicons_file_path,
        args.unimorph_dicts_folder,
        args.batch_size,
        is_fairseq_pretrained=args.use_pretrained_fairseq_model,
    )

    outputs = morpheusNMT.morph(
        args.lang,
        languages.get(part3=args.lang).alpha2,
        source_sents,
        target_sents,
        source_sents_tokenized,
        source_sents_lemma,
        source_sents_msd,
        out_path=f"{args.logs_folder}/candidates_{args.lang}.txt",
        use_chrf=args.use_chrf,
        max_count=args.max_count,
    )

    with open(os.path.join(args.outputs_folder, src_file_name + ".adv"), "w") as out_stream, open(
            os.path.join(args.outputs_folder, src_file_name + ".adv_info"), "w"
    ) as out_info_stream:

        for (stext, strans, (adv, adv_pred, orig_pred, adv_bleu, orig_bleu, is_perturbed),) in zip(
                source_sents, target_sents, outputs):
            out_stream.write(f"{adv} ||| {strans}" + "\n")
            if is_perturbed:
                n_new += 1
                out_info_stream.write(
                    f"Perturbed\n{orig_bleu:.1f} ||| {stext} ||| {orig_pred} ||| "
                    f"{strans}\n{adv_bleu:.1f} ||| {adv} ||| {adv_pred} ||| {strans}\n\n"
                )
            else:
                out_info_stream.write(
                    f"Not Perturbed\n{orig_bleu:.1f} ||| {stext} ||| {orig_pred} ||| "
                    f"{strans}\n{adv_bleu:.1f} ||| {adv} ||| {adv_pred} ||| {strans}\n\n"
                )

    logger.info(f"complete. n_new/n_total: {n_new}/{len(sentences_meta)}")
