import logging
import os
import random
from typing import List

import sacrebleu
import torch
from fairseq.models.transformer import TransformerModel
from sacremoses import MosesDetokenizer
from tqdm import tqdm

from .morpheus_base import MorpheusBase

"""
Implements `morph` and `search` method for the MT task. Still an abstract class since 
some key methods will vary depending on the target model.
"""


class MorpheusNMT(MorpheusBase):
    def __init__(
            self, lang, lang_short, reinflection_lexicons_folder, unimorphs_dicts_folder
    ):
        super(MorpheusNMT, self).__init__(
            reinflection_lexicons_folder, unimorphs_dicts_folder, lang, lang_short
        )

    def morph(
            self,
            lang: str,
            lang_short: str,
            source_sents: List[str],
            target_sents: List[str],
            source_sents_tokenized: List[List[str]],
            source_sents_lemma: List[List[str]],
            source_sents_msd: List[List[str]],
            out_path: str,
            constrain_pos: bool = True,
            use_chrf: bool = False,
            max_count: int = 50
    ):

        MAX_COUNT = max_count
        # get inflections for all sentences
        DOC_SIZE = len(source_sents)
        logging.debug(f"total sentences in document: {DOC_SIZE}")

        logging.debug("translating original data")
        orig_scores, orig_predicted = self.get_score_batch(
            source_sents, target_sents, use_chrf=use_chrf
        )
        logging.debug("finished translating original data")

        candidate_sents = []
        candidate_indices = []
        references = []
        max_inflections = 0
        logging.debug(f"extracting candidate perturbations")
        logging.debug(f"pooling max {MAX_COUNT} perturbations per sentence")

        for sent_idx in tqdm(range(DOC_SIZE)):
            orig_tokenized = source_sents_tokenized[sent_idx]
            ref = target_sents[sent_idx]

            token_inflections = self.get_inflections(
                orig_tokenized,
                source_sents_lemma[sent_idx],
                source_sents_msd[sent_idx],
                constrain_pos,
            )

            max_inf = 1
            for x in token_inflections:
                max_inf *= len(x[1])
            max_inflections += max_inf

            source_perturbations = set()
            for _ in range(MAX_COUNT):
                perturbed_sent = []
                for _, infls in token_inflections:
                    sampled_token = random.sample(infls, 1)[0]
                    perturbed_sent.append(sampled_token)
                perturbed_sent = MosesDetokenizer(lang=lang_short).detokenize(perturbed_sent)
                source_perturbations.add(perturbed_sent)

            candidate_sents.extend(list(source_perturbations))
            candidate_indices.extend([sent_idx] * len(source_perturbations))
            references.extend([ref] * len(source_perturbations))

        logging.debug(f"size of perturbed sentence pool: {len(candidate_sents)}")
        logging.debug(f"pooled {len(candidate_sents) / DOC_SIZE :.2f} perturbations per sentence")
        logging.debug(
            f"oracle poolable {max_inflections / DOC_SIZE :.2f} perturbations per sentence")

        logging.debug(f"starting translation of perturbations")

        chunk_size = 10000
        if len(candidate_sents) > chunk_size:  # lang == "tur"
            logging.debug(
                f"due to large pool size, perturbations pool is split into chunks for translation")
            chunks_cand_sents = [candidate_sents[i:i + chunk_size] for i in
                                 range(0, len(candidate_sents), chunk_size)]
            chunks_refs = [references[i:i + chunk_size] for i in
                           range(0, len(references), chunk_size)]
            perturbed_scores, perturbed_predicted = [], []
            for ii, (chunk_cand_sents, chunk_refs) in enumerate(
                    zip(chunks_cand_sents, chunks_refs)):
                perturbed_scores_, perturbed_predicted_ = self.get_score_batch(
                    chunk_cand_sents, chunk_refs, use_chrf=use_chrf
                )
                perturbed_scores.extend(perturbed_scores_)
                perturbed_predicted.extend(perturbed_predicted_)
                logging.debug(
                    f"translated perturbations in the range {ii * chunk_size} - {(ii + 1) * chunk_size}")
        else:
            perturbed_scores, perturbed_predicted = self.get_score_batch(
                candidate_sents, references, use_chrf=use_chrf
            )

        logging.debug(f"finished translating the candidate perturbations!")

        logging.debug(f"finding adversarial perturbations")
        idx2argmin = {}
        idx2minscore = {}
        for cand_idx in range(len(candidate_sents)):
            sent_idx = candidate_indices[cand_idx]
            score_perturbed = perturbed_scores[cand_idx]
            score_original = orig_scores[sent_idx]

            if score_perturbed < score_original:
                if sent_idx not in idx2minscore:
                    idx2minscore[sent_idx] = score_perturbed
                    idx2argmin[sent_idx] = cand_idx
                else:
                    if score_perturbed < idx2minscore[sent_idx]:
                        idx2minscore[sent_idx] = score_perturbed
                        idx2argmin[sent_idx] = cand_idx
        logging.debug(
            f"found adversarial perturbations for {len(idx2minscore)}/{DOC_SIZE} sentences"
        )

        outputs = []
        for sent_idx in range(DOC_SIZE):
            is_perturbed = sent_idx in idx2minscore
            if is_perturbed:
                outputs.append(
                    (
                        candidate_sents[idx2argmin[sent_idx]],
                        perturbed_predicted[idx2argmin[sent_idx]],
                        orig_predicted[sent_idx],
                        perturbed_scores[idx2argmin[sent_idx]],
                        orig_scores[sent_idx],
                        is_perturbed,
                    )
                )
            else:
                outputs.append(
                    (
                        source_sents[sent_idx],
                        orig_predicted[sent_idx],
                        orig_predicted[sent_idx],
                        orig_scores[sent_idx],
                        orig_scores[sent_idx],
                        is_perturbed,
                    )
                )

        logging.debug(f"writing the candidate perturbations to {out_path}")
        with open(out_path, "w") as wf:
            for cand_idx in range(len(candidate_sents)):
                sent_idx = candidate_indices[cand_idx]
                score_perturbed = perturbed_scores[cand_idx]
                score_original = orig_scores[sent_idx]

                wf.write(f"{sent_idx}\n")
                wf.write(f"Original score: {score_original:.1f}\n")
                wf.write(
                    f"{source_sents[sent_idx]} ||| {orig_predicted[sent_idx]} ||| {target_sents[sent_idx]}\n"
                )
                wf.write(f"Perturbed score: {score_perturbed:.1f}\n")
                wf.write(
                    f"{candidate_sents[cand_idx]} ||| {perturbed_predicted[cand_idx]} ||| {target_sents[sent_idx]}\n\n"
                )

        return outputs

    def get_score(self, source: str, reference: str, beam=5, use_chrf=False):
        predicted = self.model.translate(source, beam)
        if use_chrf:
            return (
                sacrebleu.sentence_chrf(predicted, [reference]).score * 100,
                predicted,
            )
        else:
            return sacrebleu.sentence_bleu(predicted, [reference]).score, predicted

    def get_score_batch(
            self, source: List[str], reference: List[str], beam=5, use_chrf=False
    ):
        predicted = self.model.translate(source, beam)
        if use_chrf:
            scores = [
                sacrebleu.sentence_chrf(pred, [ref]).score * 100
                for pred, ref in zip(predicted, reference)
            ]
        else:
            scores = [
                sacrebleu.sentence_bleu(pred, [ref]).score
                for pred, ref in zip(predicted, reference)
            ]
        return scores, predicted


"""
Implements model-specific details.
"""


class MorpheusFairseqTransformerNMT(MorpheusNMT):
    def __init__(
            self,
            model_dir,
            lang,
            lang_short,
            reinflection_lexicons_folder,
            unimorphs_dicts_folder,
            use_cuda=True,
            batch_size=4,
            is_fairseq_pretrained=False
    ):
        super(MorpheusFairseqTransformerNMT, self).__init__(
            lang, lang_short, reinflection_lexicons_folder, unimorphs_dicts_folder
        )

        if not is_fairseq_pretrained:
            self.model = TransformerModel.from_pretrained(
                model_dir,
                "checkpoint_best.pt",
                tokenizer="moses",
                bpe="sentencepiece",
                sentencepiece_model=os.path.join(model_dir, "spm8000.model"),
                max_sentences=batch_size,
            )
        else:
            assert lang in ["deu", "rus"]
            fairseq_model_name = 'transformer.wmt19.de-en' if lang == "deu" else 'transformer.wmt19.ru-en'
            print(fairseq_model_name)
            self.model = torch.hub.load(
                'pytorch/fairseq',
                fairseq_model_name,
                checkpoint_file='model1.pt',  # 'model1.pt:model2.pt:model3.pt:model4.pt',
                tokenizer='moses',
                bpe='fastbpe',
                max_sentences=batch_size)

        self.model.eval()

        if use_cuda and torch.cuda.is_available():
            self.model.cuda()
