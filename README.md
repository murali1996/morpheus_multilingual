# Morphological Analysis of Machine Translation

We extend the original ACL 2020 work (Morpheus) by Tan et al., to allow for multiple source
languages.

## Setup

First, download the package and install the requirements:

```bash
git clone https://github.com/murali1996/morpheus_multilingual
cd morpheus_multilingual
pip install -e .
```

The required packages are listed in _requirements.txt_ and can also be installed
using `pip install -r requirements.txt`

Then, install fairseq package:

```bash
pip install fairseq fastBPE
```

If you face any issues w/ fastBPE installation, see [here](https://github.com/glample/fastBPE)
or [here](https://github.com/pytorch/fairseq/issues/1224#issuecomment-539560804) to resolve issues
for fastBPE installation and complete installing fairseq.

Second, prepare the language-specific tools,

```bash
mkdir -p resources

# setup unimorph_inflect
git clone https://github.com/antonisa/unimorph_inflect.git ./resources/unimorph_inflect
cd resources/unimorph_inflect
python3 setup.py install
cd ../../

# download ud-compatibility for format conversion from stanza to unimorph
git clone https://github.com/unimorph/ud-compatibility ./resources/ud-compatibility
sed -i 's/from\ collections\.abc\ import\ Set/from\ typing\ import\ Set/g' resources/ud-compatibility/UD_UM/utils.py
# sed -i '' 's/from\ collections\.abc\ import\ Set/from\ typing\ import\ Set/g' resources/ud-compatibility/UD_UM/utils.py # for macOS

# download language specific resources
python run_setup.py deu
# can pass any number of languages
# python run_setup.py deu rus ukr
```

Irrespective of number of data files you would like to create adversaries for, you only need to
run `run_setup.py` only once per language.

### Obtaining adversarial evaluation sets

To obtain adversarial evaluation sets on the provided [sample MT data](sample_mt_data), follow the steps
below.

``` bash
# Use pretrained NMT models from fairseq to evaluate their morphological robustness on a 
# sample of newstest2018 evaluation suite from WMT

python run_create_adversaries.py -l deu -f "./sample_mt_data/deu-eng/newstest2018-dev-sample.orig.deu-eng" -m "" --use_pretrained_fairseq_model -b 4
```

## (Optional) Observe stanza outputs

To observe stanza outputs and obtain the tagged Morpho-Syntactic Descriptions (MSDs) for your MT
file's source text, follow the steps below. Run the command as-is to obtain results on the
provided [sample MT data](sample_mt_data).

Given source language text file, we first collect its vocabulary and then generate all possible
morphological forms. We use `Stanza` to generate the morpho-syntactic description (MSD) for each
token in the source text. Then, we use `ud-comptability` toolkit to map the UD-style MSD to UniMorph
format. Finally, we make use of UniMorph dictionaries and `unimorph_inflect` toolkit to generate all
possible morphological forms.

```bash 
# generate seed set for adversarial replacement
python -W ignore morpheus_multilingual/utils/create_dictionary.py -l deu -f "./sample_mt_data/deu-eng/newstest2018-dev-sample.orig.deu-eng"
```

- This might take anywhere between few seconds and few minutes to run,
- And creates 4 files in the output folder (defaults to folder of the mt_file provided above)
  with the following extensions:
    - `.stanza` (stanza processed file),
    - `.married` (stanza processed file with MSDs converted to UniMorph format),
    - `.dict` (all the unique combinations of (lemma, set(MSD)) with set(MSD) containing one
      of ["V", "N", "ADJ"]),
    - `.dict_frequency` (the observed frequency of the unique combinations).
- If your source side texts are already tokenized, you can run the above command with
  flag `--pretokenized` to disable any tokenization by Stanza
- To use a GPU while running Stanza, use the flag `--use_gpu`

## (Optional) Observe reinflection candidates

Once all the MSDs from the MT data file are obtain, the next step is to observe what can be the
potential replacement candidates (aka. reinflections) given the lemma and their MSD. These
reinflections are used in finding adversarial examples.

```bash 
# generate candidate set for adversarial replacement
python -W ignore morpheus_multilingual/utils/create_reinflections.py -l deu -f "./sample_mt_data/deu-eng/candidates/newstest2018-dev-sample.orig.deu-eng.stanza.married.dict"
```

- This run might take a bit longer than the one to observe stanza outputs
- And creates one file consisting of reinflections

## (Optional) Downloading TED-corpus data

```bash
python run_download_ted_data.py
```

## Extending to New Languages

## Citation

If you use this tool in your research, consider adding the following citation,

```bib
@inproceedings{jayanthi-pratapa-2021-study,
    title = "A Study of Morphological Robustness of Neural Machine Translation",
    author = "Jayanthi, Sai Muralidhar  and
      Pratapa, Adithya",
    booktitle = "Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.sigmorphon-1.6",
    doi = "10.18653/v1/2021.sigmorphon-1.6",
    pages = "49--59"
}
```

We also recommend citing the original MORPHEUS work from ACL 2020 that formed the basis of this
tool,

```bib
@inproceedings{tan-etal-2020-morphin,
    title = "It{'}s Morphin{'} Time! {C}ombating Linguistic Discrimination with Inflectional Perturbations",
    author = "Tan, Samson  and
      Joty, Shafiq  and
      Kan, Min-Yen  and
      Socher, Richard",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.263",
    pages = "2920--2935",
}
```
