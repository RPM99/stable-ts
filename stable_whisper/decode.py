from typing import TYPE_CHECKING, List, Union, Optional, Callable, Tuple
from dataclasses import replace
import warnings

import torch
import numpy as np

from whisper.decoding import DecodingTask, DecodingOptions, DecodingResult


if TYPE_CHECKING:
    from whisper.model import Whisper


def _suppress_ts(ts_logits: torch.Tensor, ts_token_mask: torch.Tensor = None):
    if ts_token_mask is not None:
        ts_logits[:, ts_token_mask] = -np.inf


# modified version of whisper.decoding.DecodingTask
class DecodingTaskStable(DecodingTask):

    def __init__(self, *args, **kwargs):
        self.ts_token_mask: torch.Tensor = kwargs.pop('ts_token_mask', None)
        self.audio_features: torch.Tensor = kwargs.pop('audio_features', None)
        self.keyword_spotting: Optional[Callable] = kwargs.pop('keyword_spotting', None)
        self.input_segment: Optional[Callable] = kwargs.pop('input_segment', None)
        super(DecodingTaskStable, self).__init__(*args, **kwargs)

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            if len(prefix_tokens) != 0:
                warnings.warn("DecodingTaskStablePBA: a prefix was provided but was ignored")
                prefix_tokens = []
            tokens = tokens + prefix_tokens

        if (keyword_spotting := self.keyword_spotting) != None and (segment_input := self.input_segment) != None:
            # obtain tokens corresponding to identified keywords
            biasing_prompt = keyword_spotting(input_features=segment_input)[0]
            keywords_tokens = self.tokenizer.encode(" " + biasing_prompt.strip()) if biasing_prompt.strip() != '' else []
            keywords_tokens = keywords_tokens[-min(len(keywords_tokens), (self.n_ctx // 2 - 1) * 3 // 4):].tolist() if not isinstance(keywords_tokens, list) else keywords_tokens
        else:
            keywords_tokens = []

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
        else:
            prompt_tokens = []

        if len(keywords_tokens) != 0 or len(prompt_tokens) != 0:
            tokens = (
                [self.tokenizer.sot_prev]
                + keywords_tokens
                + prompt_tokens[-((self.n_ctx // 2 - 1) - len(keywords_tokens)) :]
                + tokens
            )

        return tuple(tokens)

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
                  keyword_spotting: Optional[Callable] = None,
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
    keyword_spotting : Callable, optional
        A callable function that returns a biasing prompt with the keywords present in a given 
        segment of audio.

    Returns
    -------
    whisper.decode.DecodingResult or list whisper.decode.DecodingResult
        The result(s) of decoding contained in ``whisper.decode.DecodingResult`` dataclass instance(s).
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    task = DecodingTaskStable(model, options, ts_token_mask=ts_token_mask, audio_features=audio_features, keyword_spotting=keyword_spotting, input_segment=mel)
    result = task.run(mel)

    return result[0] if single else result, task.audio_features
