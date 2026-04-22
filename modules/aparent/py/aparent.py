#!/usr/bin/env python3

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

__author__ = "Alejandro Gonzales-Irribarren"
__email__ = "alejandrxgzi@gmail.com"
__github__ = "https://github.com/alejandrogzi"
__version__ = "0.0.6"

import argparse
import os
import numpy as np
from typing import List, Tuple, Callable
from pathlib import Path
from keras.models import load_model
from scipy.signal import correlate as sp_corr
from scipy.signal import find_peaks


EncoderType = Callable[[List[str]], List[np.ndarray]]

SEQUENCE_STRIDE = 10
CONV_SMOOTHING = True
PEAK_MIN_HEIGHT = 0.001
PEAK_THRESHOLD = 0.01
PEAK_MIN_DISTANCE = 3
PEAK_PROMINENCE = (0.01, None)
LIB_BIAS = 4
MODEL = "model/aparent_large_lessdropout_all_libs_no_sampleweights.h5"
MODEL_PATH = Path(__file__).parent / MODEL


def run() -> None:
    """
    Run APARENT to estimate poly(A) tail length from a chunked file

    Example
    -------
    >>> run()
    """
    args = parse()

    if args.model:
        model = load_model(args.model)
    else:
        model = load_model(MODEL_PATH)

    encoder = get_aparent_encoder(lib_bias=LIB_BIAS)

    (bedgraph_forward, bedgraph_reverse, bed) = process_chunk(args, model, encoder)
    write_results(
        bedgraph_forward,
        bedgraph_reverse,
        bed,
        args.outdir,
        args.prefix,
        args.mode == "bedgraph",
    )


def process_chunk(
    args: argparse.Namespace, model: str, encoder: EncoderType
) -> Tuple[List[str], List[str], List[str]]:
    """Process chunk file to estimate poly(A) tail length.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    model : str
        Path to APARENT model
    encoder : EncoderType
        Encoder function for sequences

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (bedgraph_forward, bedgraph_reverse, bed_lines)

    Example
    -------
    >>> process_chunk(args, model, encoder)
    ['chr1\t100\t101\t0.5\n']
    """
    print("INFO: processing chunk: " + args.bed)
    graph_lines_forward = []
    graph_lines_reverse = []
    bed_lines = []

    for row in open(args.bed):
        fields = row.strip().split("\t")

        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        # name = fields[3]
        strand = fields[3]
        seq = fields[4]

        peak_ixs, polya_profile = run_aparent(model, encoder, seq)

        if strand == "-":
            # INFO: reverse polya_profile and store length
            polya_profile = polya_profile[::-1]
            length = len(polya_profile)

            for i, peak in enumerate(polya_profile):
                if peak < args.threshold:
                    length = length - 1
                    # peak = 0
                    continue  # WARN: ignoring peaks below threshold

                if args.mode == "bedgraph":
                    graph_lines_reverse.append(
                        # f"{chrom}\t{end - i - 1}\t{end - i}\t{peak}\t{strand}\n"
                        f"{chrom}\t{end - length}\t{end - length + 1}\t{peak}\n"
                    )
                elif args.mode == "bed":
                    line = [
                        chrom,
                        end - length,
                        end - length + 1,
                        f"peak_{i}",
                        peak,
                        strand,
                    ]
                    bed_lines.append("\t".join(map(str, line)) + "\n")

                length = length - 1
        else:
            for i, peak in enumerate(polya_profile):
                if peak < args.threshold:
                    # peak = 0
                    continue

                if args.mode == "bedgraph":
                    graph_lines_forward.append(
                        # f"{chrom}\t{start + i - 1}\t{start + i}\t{peak}\t{strand}\n"
                        f"{chrom}\t{start + i - 1}\t{start + i}\t{peak}\n"
                    )
                elif args.mode == "bed":
                    line = [
                        chrom,
                        start + i - 1,
                        start + i,
                        f"peak_{i}",
                        peak,
                        strand,
                    ]
                    bed_lines.append("\t".join(map(str, line)) + "\n")

        if args.use_max_peak:
            all_peak_ixs = peak_ixs
            peak_ixs = [np.argmax(polya_profile)]

            if (
                len(all_peak_ixs) != 1 or peak_ixs[0] not in all_peak_ixs
            ) and args.verbose:
                print(
                    f"peak has divergent peaks / max peaks:\t{all_peak_ixs}\t{peak_ixs}"
                )

    print("INFO: finished processing chunk: " + args.bed)
    return (graph_lines_forward, graph_lines_reverse, bed_lines)


def write_results(
    bedgraph_forward: List[str],
    bedgraph_reverse: List[str],
    bed_lines: List[str],
    outdir: str,
    prefix: str,
    bg_flag: bool,
) -> None:
    """Write results to output files.

    Parameters
    ----------
    bedgraph_forward : List[str]
        Forward strand bedgraph lines
    bedgraph_reverse : List[str]
        Reverse strand bedgraph lines
    bed_lines : List[str]
        BED format lines
    outdir : str
        Output directory path
    prefix : str
        Prefix for output filenames
    bg_flag : bool
        Write bedgraph format if True, BED format if False

    Example
    -------
    >>> write_results(["chr1\t100\t101\t0.5\n"], [], [], "output", "sample", True)
    """
    os.makedirs(outdir, exist_ok=True)
    print("INFO: writing results to: " + outdir)

    if bg_flag:
        bg_forward = f"{outdir}/{prefix}.aparent.forward.bg"
        bg_reverse = f"{outdir}/{prefix}.aparent.reverse.bg"

        if len(bedgraph_forward) > 0:
            print("INFO: writing bedgraph to: " + bg_forward)
            with open(bg_forward, "w") as f:
                f.writelines(bedgraph_forward)

        if len(bedgraph_reverse) > 0:
            print("INFO: writing bedgraph to: " + bg_reverse)
            with open(bg_reverse, "w") as f:
                f.writelines(bedgraph_reverse)
    else:
        bed = f"{outdir}/{prefix}.aparent.bed"
        print("INFO: writing bed to: " + bed)

        if len(bed_lines) > 0:
            with open(bed, "w") as f:
                f.writelines(bed_lines)

    return None


def run_aparent(
    model: str, encoder: EncoderType, seq: str
) -> Tuple[List[int], List[float]]:
    """Run APARENT to predict poly(A) peaks for a sequence.

    Parameters
    ----------
    model : str
        Path to APARENT Keras model
    encoder : EncoderType
        Encoder function for sequences
    seq : str
        DNA sequence to analyze

    Returns
    -------
    Tuple[List[int], List[float]]
        Peak indices and poly(A) signal profile

    Example
    -------
    >>> peaks, profile = run_aparent(model, encoder, "ACGT...")
    >>> len(peaks)
    3
    """
    peak_ixs, polya_profile = find_polya_peaks(
        model,
        encoder,
        seq,
        sequence_stride=SEQUENCE_STRIDE,
        conv_smoothing=CONV_SMOOTHING,
        peak_min_height=PEAK_MIN_HEIGHT,
        peak_min_distance=PEAK_MIN_DISTANCE,
        peak_prominence=PEAK_PROMINENCE,
    )

    return (peak_ixs, polya_profile)


class OneHotEncoder:
    """
    One-hot encoder for DNA sequences

    Encodes DNA sequences into one-hot numpy arrays for use with
    neural network models.

    Parameters
    ----------
    seq_length : int, optional
        Length of sequences to encode (default: 100)
    default_fill_value : float, optional
        Value to fill when encountering unknown bases (default: 0)

    Example
    -------
    >>> encoder = OneHotEncoder(205)
    >>> encoded = encoder("ACGTACGT")
    >>> print(encoded.shape)
    (205, 4)
    """

    def __init__(self, seq_length=100, default_fill_value=0):
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value
        self.encode_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.decode_map = {0: "A", 1: "C", 2: "G", 3: "T", -1: "X"}

    def encode(self, seq):
        """
        Encode a DNA sequence to one-hot array

        Parameters
        ----------
        seq : str
            DNA sequence string

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (seq_length, 4)

        Example
        -------
        >>> encoder = OneHotEncoder(205)
        >>> result = encoder.encode("ACGT")
        >>> print(result.shape)
        (205, 4)
        """
        one_hot = np.zeros((self.seq_length, 4))
        self.encode_inplace(seq, one_hot)

        return one_hot

    def encode_inplace(self, seq, encoding):
        """
        Encode a DNA sequence into a pre-allocated array

        Parameters
        ----------
        seq : str
            DNA sequence string
        encoding : np.ndarray
            Pre-allocated array of shape (seq_length, 4)

        Returns
        -------
        None

        Example
        -------
        >>> encoder = OneHotEncoder(205)
        >>> arr = np.zeros((205, 4))
        >>> encoder.encode_inplace("ACGT", arr)
        """
        for pos, nt in enumerate(list(seq)):
            if nt in self.encode_map:
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0:
                encoding[pos, :] = self.default_fill_value

    def __call__(self, seq):
        """
        Encode a DNA sequence to one-hot array

        Parameters
        ----------
        seq : str
            DNA sequence string

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (seq_length, 4)

        Example
        -------
        >>> encoder = OneHotEncoder(205)
        >>> result = encoder("ACGT")
        >>> print(result.shape)
        (205, 4)
        """
        return self.encode(seq)


def logit(x):
    """
    Compute the logit (log-odds) function

    Parameters
    ----------
    x : float
        Probability value between 0 and 1

    Returns
    -------
    float
        Log-odds value

    Example
    -------
    >>> logit(0.5)
    0.0
    >>> logit(0.9)
    2.1972245773362196
    """
    return np.log(x / (1.0 - x))


def get_aparent_encoder(lib_bias=None):
    """
    Get APARENT encoder with 205bp sequence length

    Returns an encoder function for preparing sequences for the
    APARENT model with 205bp input length.

    Parameters
    ----------
    lib_bias : int, optional
        Library bias index (default: None)

    Returns
    -------
    Callable
        Encoder function that takes a list of sequences and returns
        [one_hots, fake_lib, fake_d] arrays

    Example
    -------
    >>> encoder = get_aparent_encoder(lib_bias=4)
    >>> result = encoder(["ACGTACGT"])
    >>> print(len(result))
    3
    """
    onehot_encoder = OneHotEncoder(205)

    def encode_for_aparent(sequences):
        one_hots = np.concatenate(
            [
                np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1))
                for sequence in sequences
            ],
            axis=0,
        )

        fake_lib = np.zeros((len(sequences), 13))
        fake_d = np.ones((len(sequences), 1))

        if lib_bias is not None:
            fake_lib[:, lib_bias] = 1.0

        return [one_hots, fake_lib, fake_d]

    return encode_for_aparent


def get_aparent_legacy_encoder(lib_bias=None):
    """
    Get legacy APARENT encoder with 185bp sequence length

    Returns an encoder function for preparing sequences for the
    legacy APARENT model with 185bp input length.

    Parameters
    ----------
    lib_bias : int, optional
        Library bias index (default: None)

    Returns
    -------
    Callable
        Encoder function that takes a list of sequences and returns
        [one_hots, fake_lib, fake_d] arrays

    Example
    -------
    >>> encoder = get_aparent_legacy_encoder(lib_bias=4)
    >>> result = encoder(["ACGTACGT"])
    >>> print(len(result))
    3
    """
    onehot_encoder = OneHotEncoder(185)

    def encode_for_aparent(sequences):
        one_hots = np.concatenate(
            [
                np.reshape(onehot_encoder(sequence), (1, 1, len(sequence), 4))
                for sequence in sequences
            ],
            axis=0,
        )

        fake_lib = np.zeros((len(sequences), 36))
        fake_d = np.ones((len(sequences), 1))

        if lib_bias is not None:
            fake_lib[:, lib_bias] = 1.0

        return [one_hots, fake_lib, fake_d]

    return encode_for_aparent


def get_apadb_encoder():
    """
    Get APADB encoder for poly(A) site prediction

    Returns an encoder function for preparing sequences for the
    APADB model with proximal and distal sequence pairs.

    Returns
    -------
    Callable
        Encoder function that takes sequences and cut positions
        and returns encoded arrays

    Example
    -------
    >>> encoder = get_apadb_encoder()
    >>> result = encoder(
    ...     prox_sequences=["ACGTACGT"],
    ...     dist_sequences=["ACGTACGT"],
    ...     prox_cut_starts=[10],
    ...     prox_cut_ends=[20],
    ...     dist_cut_starts=[30],
    ...     dist_cut_ends=[40],
    ...     site_distances=[100],
    ... )
    >>> print(len(result))
    9
    """
    onehot_encoder = OneHotEncoder(205)

    def encode_for_apadb(
        prox_sequences,
        dist_sequences,
        prox_cut_starts,
        prox_cut_ends,
        dist_cut_starts,
        dist_cut_ends,
        site_distances,
    ):
        prox_one_hots = np.concatenate(
            [
                np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1))
                for sequence in prox_sequences
            ],
            axis=0,
        )
        dist_one_hots = np.concatenate(
            [
                np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1))
                for sequence in dist_sequences
            ],
            axis=0,
        )

        return [
            prox_one_hots,
            dist_one_hots,
            np.array(prox_cut_starts).reshape(-1, 1),
            np.array(prox_cut_ends).reshape(-1, 1),
            np.array(dist_cut_starts).reshape(-1, 1),
            np.array(dist_cut_ends).reshape(-1, 1),
            np.log(np.array(site_distances).reshape(-1, 1)),
            np.zeros((len(prox_sequences), 13)),
            np.ones((len(prox_sequences), 1)),
        ]

    return encode_for_apadb


def find_polya_peaks(
    aparent_model,
    aparent_encoder,
    seq,
    sequence_stride=10,
    conv_smoothing=True,
    peak_min_height=0.01,
    peak_min_distance=50,
    peak_prominence=(0.01, None),
):
    """
    Find poly(A) peaks in a DNA sequence using APARENT model

    Uses a sliding window approach to predict poly(A) signal locations
    in a DNA sequence and identifies peaks in the prediction profile.

    Parameters
    ----------
    aparent_model : keras.Model
        Loaded APARENT model
    aparent_encoder : callable
        Encoder function for preparing sequences
    seq : str
        DNA sequence to analyze
    sequence_stride : int, optional
        Stride for sliding window (default: 10)
    conv_smoothing : bool, optional
        Apply convolutional smoothing (default: True)
    peak_min_height : float, optional
        Minimum peak height (default: 0.01)
    peak_min_distance : int, optional
        Minimum distance between peaks (default: 50)
    peak_prominence : tuple, optional
        Peak prominence range (default: (0.01, None))

    Returns
    -------
    Tuple[List[int], List[float]]
        List of peak indices and poly(A) signal profile

    Example
    -------
    >>> encoder = get_aparent_encoder()
    >>> model = load_model("model.h5")
    >>> peaks, profile = find_polya_peaks(model, encoder, "ACGT...")
    >>> print(len(peaks))
    3
    """
    cut_pred_padded_slices = []
    cut_pred_padded_masks = []

    start_pos = 0
    end_pos = 205
    while True:
        seq_slice = ""
        effective_len = 0

        if end_pos <= len(seq):
            seq_slice = seq[start_pos:end_pos]
            effective_len = 205
        else:
            seq_slice = (seq[start_pos:] + ("X" * 200))[:205]
            effective_len = len(seq[start_pos:])

        _, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice]), verbose=0)

        # print("Striding over subsequence [" + str(start_pos) + ", " + str(end_pos) + "] (Total length = " + str(len(seq)) + ")...")

        padded_slice = np.concatenate(
            [
                np.zeros(start_pos),
                np.ravel(cut_pred)[:effective_len],
                np.zeros(len(seq) - start_pos - effective_len),
                np.array([np.ravel(cut_pred)[205]]),
            ],
            axis=0,
        )

        padded_mask = np.concatenate(
            [
                np.zeros(start_pos),
                np.ones(effective_len),
                np.zeros(len(seq) - start_pos - effective_len),
                np.ones(1),
            ],
            axis=0,
        )[: len(seq) + 1]

        cut_pred_padded_slices.append(padded_slice.reshape(1, -1))
        cut_pred_padded_masks.append(padded_mask.reshape(1, -1))

        if end_pos >= len(seq):
            break

        start_pos += sequence_stride
        end_pos += sequence_stride

    cut_slices = np.concatenate(cut_pred_padded_slices, axis=0)[:, :-1]
    cut_masks = np.concatenate(cut_pred_padded_masks, axis=0)[:, :-1]

    if conv_smoothing:
        smooth_filter = np.array(
            [
                [
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.085,
                    0.175,
                    0.3,
                    0.175,
                    0.085,
                    0.05,
                    0.025,
                    0.01,
                    0.005,
                ]
            ]
        )

        cut_slices = sp_corr(cut_slices, smooth_filter, mode="same")

    avg_cut_pred = np.sum(cut_slices, axis=0) / np.sum(cut_masks, axis=0)
    std_cut_pred = np.sqrt(
        np.sum((cut_slices - np.expand_dims(avg_cut_pred, axis=0)) ** 2, axis=0)
        / np.sum(cut_masks, axis=0)
    )

    peak_ixs, _ = find_peaks(
        avg_cut_pred,
        height=peak_min_height,
        distance=peak_min_distance,
        prominence=peak_prominence,
    )

    return peak_ixs.tolist(), avg_cut_pred


def score_polya_peaks(
    aparent_model,
    aparent_encoder,
    seq,
    peak_ixs,
    sequence_stride=2,
    strided_agg_mode="max",
    iso_scoring_mode="both",
    score_unit="log",
):
    """
    Score poly(A) peaks for isoform usage prediction

    Calculates isoform usage scores for poly(A) peaks using the
    APARENT model with multiple window positions around each peak.

    Parameters
    ----------
    aparent_model : keras.Model
        Loaded APARENT model
    aparent_encoder : callable
        Encoder function for preparing sequences
    seq : str
        DNA sequence to analyze
    peak_ixs : List[int]
        List of peak indices from find_polya_peaks
    sequence_stride : int, optional
        Stride for sliding window (default: 2)
    strided_agg_mode : str, optional
        Aggregation mode: "max", "mean", or "median" (default: "max")
    iso_scoring_mode : str, optional
        Scoring mode: "both", "from_iso", or "from_cuts" (default: "both")
    score_unit : str, optional
        Score unit: "log" or "raw" (default: "log")

    Returns
    -------
    List[float]
        List of isoform scores for each peak

    Example
    -------
    >>> encoder = get_aparent_encoder()
    >>> model = load_model("model.h5")
    >>> peaks, profile = find_polya_peaks(model, encoder, "ACGT...")
    >>> scores = score_polya_peaks(model, encoder, "ACGT...", peaks)
    >>> print(scores)
    [0.123, 0.456, 0.789]
    """
    peak_iso_scores = []

    iso_pred_dict = {}
    iso_pred_from_cuts_dict = {}

    for peak_ix in peak_ixs:
        iso_pred_dict[peak_ix] = []
        iso_pred_from_cuts_dict[peak_ix] = []

        if peak_ix > 75 and peak_ix < len(seq) - 150:
            for j in range(0, 30, sequence_stride):
                seq_slice = (("X" * 35) + seq + ("X" * 35))[
                    peak_ix + 35 - 80 - j : peak_ix + 35 - 80 - j + 205
                ]

                if len(seq_slice) != 205:
                    continue

                iso_pred, cut_pred = aparent_model.predict(
                    x=aparent_encoder([seq_slice])
                )

                iso_pred_dict[peak_ix].append(iso_pred[0, 0])
                iso_pred_from_cuts_dict[peak_ix].append(np.sum(cut_pred[0, 77:107]))

        if len(iso_pred_dict[peak_ix]) > 0:
            iso_pred = np.mean(iso_pred_dict[peak_ix])
            iso_pred_from_cuts = np.mean(iso_pred_from_cuts_dict[peak_ix])
            if strided_agg_mode == "max":
                iso_pred = np.max(iso_pred_dict[peak_ix])
                iso_pred_from_cuts = np.max(iso_pred_from_cuts_dict[peak_ix])
            elif strided_agg_mode == "median":
                iso_pred = np.median(iso_pred_dict[peak_ix])
                iso_pred_from_cuts = np.median(iso_pred_from_cuts_dict[peak_ix])

            if iso_scoring_mode == "both":
                peak_iso_scores.append((iso_pred + iso_pred_from_cuts) / 2.0)
            elif iso_scoring_mode == "from_iso":
                peak_iso_scores.append(iso_pred)
            elif iso_scoring_mode == "from_cuts":
                peak_iso_scores.append(iso_pred_from_cuts)

            if score_unit == "log":
                peak_iso_scores[-1] = np.log(
                    peak_iso_scores[-1] / (1.0 - peak_iso_scores[-1])
                )

            peak_iso_scores[-1] = round(peak_iso_scores[-1], 3)
        else:
            peak_iso_scores.append(-10)

    return peak_iso_scores


def parse() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    argparse.Namespace

    Example
    -------
    >>> parse()
    """
    parser = argparse.ArgumentParser(
        description="Run APARENT to estimate poly(A) tail length from a chunked file"
    )
    parser.add_argument(
        "-b",
        "--bed",
        type=str,
        help="Path to chunk input file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to output directory",
        default="aparent",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="Prefix for output files",
        default="peaks",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Wrapper version {__version__}",
    )
    parser.add_argument(
        "-mp",
        "--use_max_peak",
        action="store_const",
        const=True,
        metavar="use only the max peak of the APARENT frame",
    )
    parser.add_argument(
        "-M",
        "--mode",
        type=str,
        help="Output mode (bedgraph or bed)",
        choices=["bedgraph", "bed"],
        default="bedgraph",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to APARENT model",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=PEAK_THRESHOLD,
        help="Peak threshold",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run()
