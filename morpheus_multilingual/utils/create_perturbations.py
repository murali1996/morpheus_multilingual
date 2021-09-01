import os

import jsonlines
from tqdm import tqdm


def create_inflections_lookup(inflections_file_path):
    lookup = {}
    inflections = [ln.split("\t")[:3] for ln in open(inflections_file_path)]
    for items in inflections:
        _lemma, _msd, alternative = items
        _combination = f"{_lemma}\t{_msd}"
        if _combination not in lookup:
            lookup[_combination] = []
        lookup[_combination].append(alternative)
    for k, v in lookup.items():
        lookup[k] = list(set(v))
    print(f"number of keys in inflection lookup: {len(lookup)}")
    return lookup


def get_sentences_meta(stanza_married_file_path):
    s_meta = open(stanza_married_file_path).read().split("\n\n")

    if s_meta[-1] == "":
        s_meta = s_meta[:-1]

    # for every sentence_meta that doesn't start with a sentence_id, attach it to previous sentence
    s_meta_new = []
    for _meta in s_meta:
        if _meta.startswith("#"):
            s_meta_new.append(_meta)
        else:
            s_meta_new[-1] += f"\n{_meta}"
    print(len(s_meta_new))
    s_meta = s_meta_new

    return s_meta


def get_all_(possibilities, ind, all_sentences, stack):
    if ind >= len(possibilities):
        all_sentences.append(" ".join(stack))
        return
    for poss in possibilities[ind]:
        stack.append(poss)
        get_all_(possibilities, ind + 1, all_sentences, stack)
        _ = stack.pop()
    return


if __name__ == "__main__":

    """ specify paths """

    lg = "est"
    stanza_married_file = (
        f"../candidate_generation/candidates/ted-dev.orig.{lg}-eng.stanza.married"
    )
    inflections_out_file = (
        f"../inflection/outputs/ted-dev.orig.{lg}-eng.stanza.married.dict_inflected"
    )
    out_folder = "./outputs"

    name = inflections_out_file.split("/")[-1]
    perturbations_out_file = os.path.join(out_folder, name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    """ load inflections data and create a lookup dict """

    inflections_lookup = create_inflections_lookup(inflections_out_file)

    """ load stanza sentences """

    sentences_meta = get_sentences_meta(stanza_married_file)

    """ obtain meta information for each sentences, 
        then obtain all hased {lemma}\t{msd} combinations 
    """

    jsonl = []
    for meta in tqdm(sentences_meta):
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
        possibilities = (
            []
        )  # List[List[str]], for eaxh word all possibilities are obtained
        for sline in slines:
            sparts = sline.split("\t")
            org_word, lemma, msd = sparts[1], sparts[2], sparts[5]
            combination = f"{lemma}\t{msd}"
            if combination in inflections_lookup:
                possibilities.append(inflections_lookup[combination])
            else:
                possibilities.append([])
            # adding so as to produce a stanza tokenized version of `stext`
            possibilities[-1].append(org_word)
        all_sentences = []
        get_all_(possibilities, 0, all_sentences, [])
        dct = {
            "org_src": stext,
            "org_tgt": strans,
            "tok_src": all_sentences[0],
            "choices": all_sentences[1:] if len(all_sentences) > 1 else [],
            "n_choices": len(all_sentences) - 1,
        }
        jsonl.append(dct)
        # opfile = jsonlines.open(perturbations_out_file, "a")
        # opfile.write(jsonl[-1])
        # opfile.close()
        # print(len(all_sentences))
        # if len(all_sentences) > 10000:
        #     opfile = open(f"{len(all_sentences)}.txt", "w")
        #     opfile.write(stext+"\n")
        #     opfile.write(strans+"\n\n")
        #     opfile.write("\n".join(slines)+"\n\n")
        #     for val in possibilities:
        #         opfile.write(str(len(val))+"\t"+"\t".join(val)+"\n")
        #     opfile.write("\n")
        #     for line in all_sentences:
        #         opfile.write(line+"\n")
        #     opfile.close()
    opfile = jsonlines.open(perturbations_out_file, "w")
    for line in jsonl:
        opfile.write(jsonl)
    opfile.close()
    print(f"{len(jsonl)} lines written to {perturbations_out_file}")

    print("complete")
