from typing import TYPE_CHECKING, List, Union
from dataclasses import replace

import torch
import numpy as np

from whisper.decoding import DecodingTask, DecodingOptions, DecodingResult, BeamSearchDecoder


if TYPE_CHECKING:
    from whisper.model import Whisper
    
from torch import Tensor
from typing import Tuple, Optional, Iterable
import torch.nn.functional as F


def forward(
    sequence: List[int],
    f_sequence: List[int],
    i: int,
    x: int
) -> Tuple[int, bool]:
     
    full_match = False
    
    if sequence[i] == x:
        q = i + 1
        if q == len(sequence):
            full_match, q = (True, 0)
    else:
        k =  f_sequence[i]
        while k >= 0 and sequence[k] != x:
            k = f_sequence[k]
        q = k + 1
        
    return (q, full_match)


def potential(
    v: Iterable[int],
    delta: float
) -> float:
    return max([v_ * delta for v_ in v])
    
    
def ComputeBonus(
    biasing_phrases: List[Tuple[List[int], List[int]]],
    partial_matching_lengths: Tensor,
    x: Tensor ,
    delta: float = 1.0
) -> Tuple[List[int], bool]:
    
    any_match = False
    n_biasing_phrases = len(biasing_phrases)
    new_partial_matching_lengths = [0] * n_biasing_phrases
    
    for b in range(n_biasing_phrases):
        u, match = forward(biasing_phrases[b][0], biasing_phrases[b][1], partial_matching_lengths[b].item(), x.item())
        if match:
            any_match = True
            new_partial_matching_lengths[b] = len(biasing_phrases[b][0])
        else:
            new_partial_matching_lengths[b] = u
    
    bonus = potential(new_partial_matching_lengths, delta) - potential(partial_matching_lengths, delta)
    
    if any_match:
        new_partial_matching_lengths = [0] * n_biasing_phrases
        
    return (new_partial_matching_lengths, bonus)
    
    
class BeamSearchDecoderModified(BeamSearchDecoder):
    
    def __init__(self, *args, **kwargs):        
        self.biasing_phrases = kwargs.pop("biasing_phrases")
        self.kmp = (self.biasing_phrases != None)
        self.kmp_bonus = kwargs.pop("kmp_bonus")
        self.kmp_bonus = self.kmp_bonus if self.kmp_bonus != None else 1.0
        super(BeamSearchDecoderModified, self).__init__(*args, **kwargs)    
        
    
    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, partial_matching_lengths: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, next_partial_matching_lengths, finished_sequences = [], [], [], []
        for i in range(n_audio):
            scores, sources, pmls, bonuses, finished = {}, {}, {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    if self.kmp:
                        pml, bonus = ComputeBonus(self.biasing_phrases, partial_matching_lengths[j], token, self.kmp_bonus)
                    new_logprob = (sum_logprobs[idx] + logprob + bonus).item() if self.kmp else (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx
                    pmls[sequence] = pml
                    bonuses[sequence] = bonus

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence] - bonuses[sequence]
                    next_partial_matching_lengths.append(pmls[sequence])
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        partial_matching_lengths = torch.tensor(next_partial_matching_lengths, device=tokens.device).long()
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed, partial_matching_lengths


def _suppress_ts(ts_logits: torch.Tensor, ts_token_mask: torch.Tensor = None):
    if ts_token_mask is not None:
        ts_logits[:, ts_token_mask] = -np.inf


# modified version of whisper.decoding.DecodingTask
class DecodingTaskStable(DecodingTask):

    def __init__(self, *args, **kwargs):
        self.ts_token_mask: torch.Tensor = kwargs.pop('ts_token_mask', None)
        self.audio_features: torch.Tensor = kwargs.pop('audio_features', None)
        self.biasing_phrases: Optional[List[Tuple[List[int], List[int]]]] = kwargs.pop('biasing_phrases', None)
        self.kmp_bonus: float = kwargs.pop('kmp_bonus', None)
        super(DecodingTaskStable, self).__init__(*args, **kwargs)
        if self.options.beam_size != None and self.biasing_phrases != None:
            self.decoder = BeamSearchDecoderModified(
                beam_size = self.options.beam_size, eot = self.tokenizer.eot, inference = self.inference, patience = self.options.patience, biasing_phrases = self.biasing_phrases, kmp_bonus = self.kmp_bonus
            )

    def _get_audio_features(self, mel: torch.Tensor):
        if self.audio_features is None:
            audio_features = super()._get_audio_features(mel)
            self.audio_features = audio_features.detach().clone()
            return audio_features
        return self.audio_features.clone()

    # modified version of whisper.DecodingTask._main_loop
    def _main_loop(self, audio_features: torch.Tensor, tokens: torch.Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: torch.Tensor = torch.zeros(n_batch, device=audio_features.device)
        partial_matching_lengths: Optional[torch.Tensor] = torch.zeros(n_batch, len(self.biasing_phrases), device=audio_features.device).long() if self.biasing_phrases != None else None
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # suppress timestamp tokens where the audio is silent so that decoder ignores those timestamps
                _suppress_ts(logits[:, self.tokenizer.timestamp_begin:], self.ts_token_mask)

                logits.nan_to_num_(-np.inf)
                # expand the tokens tensor with the selected next tokens
                if self.options.beam_size != None and self.biasing_phrases != None:
                    tokens, completed, partial_matching_lengths = self.decoder.update(tokens, logits, sum_logprobs, partial_matching_lengths)
                else:
                    tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs


# modified version of whisper.decoding.decode
@torch.no_grad()
def decode_stable(model: "Whisper",
                  mel: torch.Tensor,
                  options: DecodingOptions = DecodingOptions(),
                  ts_token_mask: torch.Tensor = None,
                  audio_features: torch.Tensor = None,
                  biasing_phrases: Optional[List[Tuple[List[int], List[int]]]] = None,
                  kmp_bonus: Optional[float] = None,
                  **kwargs, ) -> \
        Union[DecodingResult, List[DecodingResult], tuple]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    mel : torch.Tensor,
        A tensor containing the Mel spectrogram(s). ``mel.shape`` must be (80, 3000) or (*, 80, 3000).
    options : whisper.decode.DecodingOptions, default whisper.decode.DecodingOptions()
        A dataclass that contains all necessary options for decoding 30-second segments
    ts_token_mask : torch.Tensor, optional
        Mask for suppressing to timestamp token(s) for decoding.
    audio_features : torch.Tensor, optional
        Reused ``audio_feature`` from encoder for fallback.

    Returns
    -------
    whisper.decode.DecodingResult or list whisper.decode.DecodingResult
        The result(s) of decoding contained in ``whisper.decode.DecodingResult`` dataclass instance(s).
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    task = DecodingTaskStable(model, options, ts_token_mask=ts_token_mask, audio_features=audio_features, biasing_phrases=biasing_phrases, kmp_bonus=kmp_bonus)
    result = task.run(mel)

    return result[0] if single else result, task.audio_features
