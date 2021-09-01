import argparse
import logging
import os
import random
import sys

import stanza
from iso639 import languages
from stanza.utils.conll import CoNLL
from tqdm import tqdm

logger = logging.getLogger(__name__)

SEED = 11737


def get_candidates(
        lang,
        ud_compatability_tool_folder,
        unimorph_dicts_folder,
        outputs_folder,
        # inputs to use only when passing ted-formatted file path
        ted_formatted_file_path=None,
        # inputs to use only when passing ted-formatted lines directly instead of file path
        tokenize_pretokenized=False,
        all_lines=[],
        all_untokenized_lines=[],
        src_file_name="",
        # <----
        use_gpu=False,
        save_dicts=True,
        sub_sample_n=-1
):
    """
    :param lang: a 3 letter standard abbreviation of the language
    :param ud_compatability_tool_folder: path consisting of marry.py file from ud-compatibility
    :param unimorph_dicts_folder: path wherein all unimorph dictionaries are stored
    :param outputs_folder: a folder where .stanza, .stanza.married, etc files are saved
    :param ted_formatted_file_path: (Optional) if provided, useed to load src and tgt sentences
    :param tokenize_pretokenized: (Optional, bool) a direct argument to Stanza pipeline
    :param all_lines: (Optional) list of src (tokenized) and tgt (||| 3-pipes seperated) tuples,
                        must be passed when `ted_formatted_file_path` is None
    :param all_untokenized_lines: (Optional) list of src (original) and tgt (||| 3-pipes seperated)
                                    tuples, must be passed along with `all_lines`
    :param src_file_name: (Optional) must be passed alongside `all_lines`, used as a file name
                            header while saving files at `outputs_folder`
    :param use_gpu: (Optional, bool) cpu or gpu device for Stanza
    :param save_dicts: (Optional, bool) If True, only then uniques lemma-MSDs are written to file
    :param sub_sample_n: (Optional, int) If given a positive integer, those many samples are
                            randomly sampled and the sampled data is saved in a file in `outputs_folder`
    :return: 3-tuple, file paths
    """

    if ted_formatted_file_path:
        assert len(all_lines) == 0, logger.error(
            "`all_lines` argument must be empty list when `ted_formatted_file_path` is not None ")
        dests = ted_formatted_file_path.strip("/").split("/")
        src_file_name = dests[-1]
        all_untokenized_lines = [line for line in open(ted_formatted_file_path)]
        all_lines = [line for line in open(ted_formatted_file_path)]
    elif len(all_lines) > 0:
        assert len(all_untokenized_lines) == len(all_lines), logger.error(
            f"len of `all_untokenized_lines` must match `all_lines`")
        assert ted_formatted_file_path is None, logger.error(
            f"`ted_formatted_file_path` must be None when `all_lines` is non empty or vice-versa")
        assert src_file_name, logger.error(
            "please provide `src_file_name` when not providing `ted_formatted_file_path`")
    else:
        raise Exception("one of `ted_formatted_file_path` or `all_lines` must be valid input")

    os.makedirs(outputs_folder, exist_ok=True)

    # https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
    short_lang_name = languages.get(part3=lang).alpha2
    logger.info(f"creating dictionary for the language: {lang}")

    untokenized_sentences = [line.split("|||")[0].strip() for line in all_untokenized_lines]
    sentences = [line.split("|||")[0].strip() for line in all_lines]
    english_translations = [line.split("|||")[1].strip() for line in all_lines]

    if sub_sample_n > 0:

        # remove all sentences wherein the src or tgt is empty
        #   this issue was observed in deu train dataset!
        inds_ = []
        for i, (x, y, z) in enumerate(zip(untokenized_sentences, sentences, english_translations)):
            if x.strip() == "" or y.strip() == "" or z.strip() == "":
                continue
            inds_.append(i)
        if len(sentences) - len(inds_) != 0:
            logger.warning(
                f"{len(sentences) - len(inds_)} number of lines have either src or tgt "
                f"as empty. Ignoring these lines.")
            untokenized_sentences = [untokenized_sentences[iii] for iii in inds_]
            sentences = [sentences[iii] for iii in inds_]
            english_translations = [english_translations[iii] for iii in inds_]

        tpls = [(x, y, z) for x, y, z in
                zip(untokenized_sentences, sentences, english_translations)]
        tpls_set = list(set(tpls))
        if len(tpls) != len(tpls_set):
            logger.warning(
                f"]There are some duplicates in the data; removing repeated "
                f"{len(tpls) - len(tpls_set)} only.")
            untokenized_sentences, sentences, english_translations = list(zip(*tpls_set))

        if sub_sample_n > len(sentences):
            logger.warning(f"You asked to sub sample {sub_sample_n} which is greater than "
                           f"total sentences available {len(sentences)}")

        inds_ = list(range(len(sentences)))
        random.seed(SEED)
        random.shuffle(inds_)
        inds_ = inds_[:sub_sample_n]

        untokenized_sentences = [untokenized_sentences[iii] for iii in inds_]
        sentences = [sentences[iii] for iii in inds_]
        english_translations = [english_translations[iii] for iii in inds_]

        # save this data
        subsample_file_name = src_file_name + f".subsampled_{sub_sample_n}"
        subsample_file_path = os.path.join(outputs_folder, subsample_file_name)
        opfile = open(subsample_file_path, "w")
        for x, y in zip(sentences, english_translations):
            opfile.write(f"{x} ||| {y}\n")
        opfile.close()

    logger.info(f"# sentences to process: {len(sentences)}")

    try:
        nlp = stanza.Pipeline(short_lang_name, processors='tokenize,pos,lemma',
                              tokenize_pretokenized=tokenize_pretokenized, use_gpu=use_gpu)
    except KeyError as e:
        logger.info(e)
        nlp = None

    if nlp:

        """part-1: obtain Stanza tags"""

        def doc2conll(stanza_doc):
            stanza_doc_dict = stanza_doc.to_dict()
            conll_ = CoNLL.convert_dict(stanza_doc_dict)
            return CoNLL.conll_as_string(conll_)

        new_lines = []
        for i, (sentence, translation) in tqdm(enumerate(zip(sentences, english_translations)),
                                               disable=logger.level != 0):
            try:
                str_ = (
                        f"# {i}\n"
                        f"# text = {untokenized_sentences[i]}\n"
                        f"# translation = {translation}\n"
                        + doc2conll(nlp(sentence))
                )
                new_lines.append(str_)
            except Exception as e:
                logger.info("")
                logger.info(f"Error: {e}")
                logger.info(f"Error: stanza couldn't produce result for {sentence}")
                continue

        stanza_file_name = src_file_name + ".stanza"
        stanza_file_path = os.path.join(outputs_folder, stanza_file_name)
        opfile = open(stanza_file_path, "w")
        for line in new_lines:
            opfile.write(line)
        opfile.close()

        """part-2: convert the tags to unimorph format"""

        logger.info("Marrying UD_UM using ud-compatability tool ...")
        married_file_name = src_file_name + ".stanza.married"
        married_file_path = os.path.join(outputs_folder, married_file_name)
        pwd = os.getcwd()
        udcomp_temp_file_path = os.path.join(ud_compatability_tool_folder, stanza_file_name)
        os.system(f"cp {stanza_file_path} {udcomp_temp_file_path}")
        os.chdir(os.path.join(pwd, ud_compatability_tool_folder))
        os.system(f"python marry.py convert --ud {stanza_file_name} -l {short_lang_name}")
        os.chdir(pwd)
        os.system(f"cp {udcomp_temp_file_path} {married_file_path}")
        os.system(f"rm {udcomp_temp_file_path}")

        """part-3: obtain all the unique combinations of (lemma, set(MSD))"""

        married_lines = [line.strip() for line in open(married_file_path).readlines()]
        uniques_dict = {}
        for line in tqdm(married_lines, disable=logger.level != 0):
            if line == "" or line.startswith("#"):
                continue
            parts = line.split("\t")
            lemma, msd = parts[2], parts[5]
            # (NEW) Stanza provides pipe seperated lemmas at times (eg. in deu lang data)
            for lem in lemma.split("|"):
                tag_ravelled = sum([x.split(".") for x in msd.split(";")], [])
                if any(x in tag_ravelled for x in ["V", "N", "ADJ"]):
                    combination = f"{lem}\t{msd}"
                    if combination not in uniques_dict:
                        uniques_dict[combination] = 0
                    uniques_dict[combination] += 1
        logger.info(f"number of keys in dict: {len(uniques_dict)}")

    else:

        """load the data and lookup for the words in the unimorph_dict"""

        logger.info(
            f"\nWARNING: For lang={lang}, for any token in ted corpus, more than once combination "
            f"of lemma and MSD \ncan exist. We retain only the last observed combination with the "
            f"hope that unimorph_inflect \nobtains all other combinations\n")
        unimorph_dict_data = {}

        if not (unimorph_dicts_folder and os.path.exists(unimorph_dicts_folder)):
            logger.info(
                f"Downloading unimorph dictionary as no stanza model exists for langauge: {lang}")
            unimorph_dicts_folder = os.path.join(outputs_folder, lang, 'unimorph_dicts')
            os.system(f"wget https://raw.githubusercontent.com/unimorph/{lang}/master/{lang} "
                      f"-P {unimorph_dicts_folder}")

        ulines = [line.strip().split("\t")
                  for line in open(os.path.join(unimorph_dicts_folder, lang)) if
                  line.strip() != ""]
        for parts in ulines:
            unimorph_dict_data.update({parts[1]: f"{parts[0]}\t{parts[2]}"})

        uniques_dict = {}
        for sentence in sentences:
            for token in sentence.split():
                if token in unimorph_dict_data:
                    combination = unimorph_dict_data[token]
                    if combination not in uniques_dict:
                        uniques_dict[combination] = 0
                    uniques_dict[combination] += 1
        logger.info(f"number of keys in dict: {len(uniques_dict)}")

        stanza_file_path = None

        married_file_name = src_file_name + ".stanza.married"
        married_file_path = os.path.join(outputs_folder, married_file_name)
        out_file_lines = []
        for i, (sentence, translation) in tqdm(enumerate(zip(sentences, english_translations)),
                                               disable=logger.level != 0):
            this_sentence_str = f"# {i}\n" \
                                f"# text = {untokenized_sentences[i]}\n" \
                                f"# translation = {translation}\n"
            for j, token in enumerate(sentence.split()):
                if token in unimorph_dict_data:
                    combination = unimorph_dict_data[token]
                    lemma_, msd_ = combination.split("\t")
                else:
                    lemma_, msd_ = "", ""
                this_token_string = f"{j}\t{token}\t{lemma_}\t\t\t{msd_}\n"
                this_sentence_str += this_token_string
            this_sentence_str += "\n"
            out_file_lines.append(this_sentence_str)
        opfile = open(married_file_path, "w")
        for line in out_file_lines:
            opfile.write(line)
        opfile.close()

    if save_dicts:

        dict_file_name = src_file_name + ".stanza.married.dict_frequency"
        dict_file_path = os.path.join(outputs_folder, dict_file_name)
        opfile = open(dict_file_path, "w")
        for str_ in [i + "\t" + str(j) + "\n" for i, j in
                     sorted(uniques_dict.items(), key=lambda item: item[1], reverse=True)]:
            opfile.write(str_)
        opfile.close()
        dict_file_name = src_file_name + ".stanza.married.dict"
        dict_file_path = os.path.join(outputs_folder, dict_file_name)
        opfile = open(dict_file_path, "w")
        for str_ in [i + "\n" for i, j in
                     sorted(uniques_dict.items(), key=lambda item: item[1], reverse=True)]:
            opfile.write(str_)
        opfile.close()

    else:

        dict_file_path = None

    responses = (stanza_file_path, married_file_path, dict_file_path)
    logger.info(f"Saved files: {responses}\n")
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using stanza, create a mapping from your data for a set of "
                    "replacement candidates"
    )
    parser.add_argument(
        "-l", "--language", required=True, type=str,
        help="3-letter language code (ISO 639-2 Code)"
    )
    parser.add_argument(
        "-f", "--mt_file_path", required=True, type=str,
        help="TED-corpus formatted file used for NMT"
    )
    parser.add_argument(
        "-o", "--outputs_folder", required=False, type=str,
        default=None,
        help="path where output files are to be dumped"
    )
    parser.add_argument(
        "-ud", "--ud_tool_folder", required=False, type=str,
        default="./resources/ud-compatibility/UD_UM",
        help="path where the UD_UM sub-directory of ud-compatibility tool's directory is located"
    )
    parser.add_argument(
        "-uni", "--unimorph_dicts_folder", required=False, type=str,
        default="./unimorph_dicts",
        help="path where the unimorph dictionaries are located"
    )
    parser.add_argument(
        "-n", "--sub_sample_n", required=False, type=int,
        default=-1,
        help="subset of data to be used through random sampling; use all if unspecified"
    )
    parser.add_argument(
        "--pretokenized", required=False,
        default=False, action="store_true",
        help="if set, stanza pipeline is called with tokenize_pretokenized=True and the inputs "
             "are not tokenized; sets tokenize_pretokenized=True to bypass the neural tokenizer"
    )
    parser.add_argument(
        "--use_gpu", required=False,
        default=False, action="store_true",
        help="if set, stanza pipeline is called with use_gpu=True"
    )

    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logger = logging.getLogger(__file__)
    # fh = logging.FileHandler('logs')
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logger.addHandler(fh)
    logger = logging.getLogger(__file__)
    if logger.level == 0:
        logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    if not args.outputs_folder:
        args.outputs_folder = os.path.join(os.path.dirname(args.mt_file_path), "candidates")
    else:
        args.outputs_folder = os.path.join("./outputs", args.mt_file_path, "candidates")

    for lang in [args.language]:
        _ = get_candidates(lang, args.ud_tool_folder, args.unimorph_dicts_folder,
                           args.outputs_folder,
                           ted_formatted_file_path=args.mt_file_path,
                           sub_sample_n=args.sub_sample_n,
                           tokenize_pretokenized=args.pretokenized, use_gpu=args.use_gpu)
