#!/usr/bin/env python3

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

"""
Takes chain or chainIDs as input.
For each chain, the script finds a gap of a certain size,
runs a local lastz job, since the resulting alignments can overlap, chains them.
Selects the best 'mini-chain' and directly adds this into the gap.
Then continues iterating.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import TypedDict

__author__ = "Ekaterina Osipova, MPI-CBG/MPI-PKS, 2018."
__credits__ = ["Bogdan M. Kirilenko", "Alejandro Gonzales-Irribarren"]
__version__ = "0.0.2"


class ChainHeader(TypedDict):
    score: int
    t_name: str
    t_start: int
    t_end: int
    q_name: str
    q_size: int
    q_strand: str
    q_start: int
    q_end: int


class GapCoordinates(TypedDict):
    block_len: int
    t_block_end: int
    t_gap_end: int
    q_block_end: int
    q_gap_end: int
    t_gap_span: int
    q_gap_span: int


class PatchRegion(TypedDict):
    t_name: str
    t_block_end: int
    t_gap_end: int
    q_name: str
    real_q_block_end: int
    real_q_gap_end: int


class ArgsNamespace(argparse.Namespace):
    chain: str
    T2bit: str
    Q2bit: str
    lastz: str
    axtChain: str
    chainSort: str
    output: str | None
    workdir: str
    verbose: bool
    chainMinScore: int
    chainMinSizeT: int
    chainMinSizeQ: int
    gapMinSizeT: int
    gapMinSizeQ: int
    gapMaxSizeT: int
    gapMaxSizeQ: int
    lastzParameters: str
    unmask: bool
    scoreThreshold: int
    index: str | None


ChainLineType = str
ChainContentType = list[str]
MiniChainBlockType = list[str]
MiniChainOutputType = tuple[ChainLineType | None, ChainContentType | None]
ShellScriptPath = str


def parse_args() -> ArgsNamespace:
    """Builds an argument parser with all required and optional arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "This script extracts a chain from all.chain file by ID, "
            "finds gaps and using lastz patches these gaps, "
            "then inserts new blocks to a chain"
        ),
        epilog=(
            "Example of use:\nchain_gap_filler.py -c hg38.speTri2.all.chain "
            "-ix hg38.speTri2.all.bb -T2 hg38.2bit "
            "-Q2 speTri2.2bit -um -m mini.chains -o out.chain"
        ),
    )
    # Required arguments
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--chain", "-c", type=str, help="all.chain file", required=True
    )
    required_named.add_argument(
        "--T2bit", "-T2", type=str, help="reference 2bit file", required=True
    )
    required_named.add_argument(
        "--Q2bit", "-Q2", type=str, help="query 2bit file", required=True
    )
    parser.add_argument(
        "--lastz",
        "-l",
        type=str,
        default="lastz",
        help="path to lastz executable, default = lastz",
    )
    parser.add_argument(
        "--axtChain",
        "-x",
        type=str,
        default="axtChain",
        help="path to axtChain executable, default = axtChain",
    )
    parser.add_argument(
        "--chainSort",
        "-s",
        type=str,
        default="chainSort",
        help="path to chainSort executable, default = chainSort",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="name of output chain file. If not specified chains go to stdout",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default="./",
        help="working directory for temp files, default = ./",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="if -v is not specified, only ERROR messages will be shown",
    )

    # Initial parameters
    parser.add_argument(
        "--chainMinScore",
        "-mscore",
        type=int,
        default=0,
        help="consider only chains with a chainMinScore, default consider all",
    )
    parser.add_argument(
        "--chainMinSizeT",
        "-mst",
        type=int,
        default=0,
        help="consider only chains with a chainMinSizeT, default consider all",
    )
    parser.add_argument(
        "--chainMinSizeQ",
        "-msq",
        type=int,
        default=0,
        help="consider only chains with a chainMinSizeQ, default consider all",
    )
    parser.add_argument(
        "--gapMinSizeT",
        "-gmint",
        type=int,
        default=10,
        help="patch only gaps that are at least that long on the target side, default gmint = 10",
    )
    parser.add_argument(
        "--gapMinSizeQ",
        "-gminq",
        type=int,
        default=10,
        help="patch only gaps that are at least that long on the query side, default gminq = 10",
    )
    parser.add_argument(
        "--gapMaxSizeT",
        "-gmaxt",
        type=int,
        default=100000,
        help="patch only gaps that are at most that long on the target side, default gmaxt = 100000",
    )
    parser.add_argument(
        "--gapMaxSizeQ",
        "-gmaxq",
        type=int,
        default=100000,
        help="patch only gaps that are at most that long on the query side, default gmaxq = 100000",
    )
    parser.add_argument(
        "--lastzParameters",
        "-lparam",
        type=str,
        default=" K=1500 L=2000 M=0 T=0 W=6 ",
        help="line with lastz parameters, default 'K=1500 L=2000 M=0 T=0 W=6' ",
    )
    parser.add_argument(
        "--unmask",
        "-um",
        action="store_true",
        help="unmasking (lower case to upper case) characters from the 2bit files",
    )
    parser.add_argument(
        "--scoreThreshold",
        "-st",
        type=int,
        default=2000,
        help="insert only chains that have at least this score, default st = 2000",
    )
    parser.add_argument("--index", "-ix", type=str, help="index.bb file for chains")
    parser.add_argument(
        "--version",
        action="version",
        version=f"repeat_filler {__version__}",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    return args


def get_chain_string(args: ArgsNamespace) -> str:
    """Extracts chains with requested ids from "all.chain" file.

    Reads the entire contents of the chain file and returns it as a string.
    The chain file format contains chain alignment records with header lines
    starting with "chain" followed by alignment blocks with gap information.

    Args:
        args: Parsed command-line arguments containing the path to the chain file.

    Returns:
        The complete contents of the chain file as a string.
    """
    with open(args.chain, "r") as content_file:
        return content_file.read()


def make_shell_list(
    input_chain: str,
    out_file: ShellScriptPath,
    args: ArgsNamespace,
) -> None:
    """Makes a list of jobs to run in temp shell script.

    Parses chain alignment data and generates shell commands to run lastz alignments
    for each gap that meets the size criteria. The generated shell script contains
    commands to patch gaps between aligned blocks with new alignments.

    Args:
        input_chain: String containing all chain alignment records from the chain file.
        out_file: Path to output shell script file; shell commands will be written here.
        args: Parsed command-line arguments containing file paths and gap filtering parameters.
    """
    out_file_handler = open(out_file, "w")

    out_file_handler.write("#!/usr/bin/env bash\n")
    out_file_handler.write("#set -o pipefail\n")
    out_file_handler.write("#set -e\n")

    gap_count: int = 0

    line_number: int = 0

    target_two_bit: str = args.T2bit
    query_two_bit: str = args.Q2bit

    lastz_arg: str = args.lastz
    axt_chain_arg: str = args.axtChain
    chain_sort_arg: str = args.chainSort

    chain_list = iter([f"{chain_line}\n" for chain_line in input_chain.split("\n")])

    for line in chain_list:
        line_number += 1

        ll: list[str] = line.split()
        if len(ll) > 0:
            if ll[0] == "chain":
                score: int = int(ll[1])
                t_name: str = ll[2]
                t_start: int = int(ll[5])
                t_end: int = int(ll[6])
                q_name: str = ll[7]
                q_size: int = int(ll[8])
                q_strand: str = ll[9]
                q_start_x: int = int(ll[10])
                q_end_x: int = int(ll[11])
                logging.info(f"q_strand = {q_strand}")

                q_start: int = q_start_x
                q_end: int = q_end_x
                lastz_parameters: str = args.lastzParameters + " --strand=plus"

                if q_strand == "-":
                    lastz_parameters = args.lastzParameters + " --strand=minus"

                if ll[4] != "+":
                    logging.error(f"ERROR: target strand is not + for chain:{line}")
                    sys.exit(1)
                logging.info(f"score of this chain = {score}")
                if (
                    (score >= args.chainMinScore)
                    and (t_end - t_start >= args.chainMinSizeT)
                    and (q_end - q_start >= args.chainMinSizeQ)
                ):
                    logging.info("valid chain")
                    current_t_position: int = t_start
                    current_q_position: int = q_start

                    line = next(chain_list)
                    line_number += 1

                    while re.match(r"^\d+", line) is not None:
                        a: list[str] = line.split()
                        if len(a) == 1:
                            logging.info("it was the last block\n")

                        else:
                            block_len: int = int(a[0])
                            t_block_end: int = current_t_position + block_len
                            q_block_end: int = current_q_position + block_len
                            t_gap_end: int = current_t_position + block_len + int(a[1])
                            q_gap_end: int = current_q_position + block_len + int(a[2])
                            t_gap_span: int = t_gap_end - t_block_end
                            q_gap_span: int = q_gap_end - q_block_end
                            if (
                                (t_gap_span >= args.gapMinSizeT)
                                and (t_gap_span <= args.gapMaxSizeT)
                                and (q_gap_span >= args.gapMinSizeQ)
                                and (q_gap_span <= args.gapMaxSizeQ)
                            ):
                                logging.info(f"yes, this gap will be patched: {line}")
                                t_block_end += 1
                                q_block_end += 1

                                unmask: str = "[unmask]" if args.unmask else ""

                                real_q_block_end: int
                                real_q_gap_end: int
                                if q_strand == "-":
                                    real_q_block_end = q_size - q_gap_end + 1
                                    real_q_gap_end = q_size - q_block_end + 1
                                else:
                                    real_q_block_end = q_block_end
                                    real_q_gap_end = q_gap_end

                                logging.info("running lastz on the block:")
                                region_to_be_patched: PatchRegion = {
                                    "t_name": t_name,
                                    "t_block_end": t_block_end,
                                    "t_gap_end": t_gap_end,
                                    "q_name": q_name,
                                    "real_q_block_end": real_q_block_end,
                                    "real_q_gap_end": real_q_gap_end,
                                }
                                command_1: str = (
                                    f"{target_two_bit}/{t_name}[{t_block_end}..{t_gap_end}]{unmask} "
                                    f"{query_two_bit}/{q_name}[{real_q_block_end}..{real_q_gap_end}]{unmask} "
                                    f"--format=axt {lastz_parameters} | "
                                )
                                command_2: str = f"-linearGap=loose stdin {target_two_bit} {query_two_bit} stdout 2> /dev/null | "
                                command_3: str = "stdin stdout"
                                command_lastz: str = (
                                    f"{lastz_arg}{command_1}{axt_chain_arg}"
                                    f"{command_2}{chain_sort_arg}{command_3}"
                                )

                                shell_command: str = (
                                    f'echo -e "LINE{line_number - 1}\\n{block_len}\\n{t_block_end}\\n{t_gap_end}\\n'
                                    f'{real_q_block_end}\\n{real_q_gap_end}\\n"; {command_lastz}; '
                                    f'echo -e "LINE{line_number - 1}\\n"\n'
                                )

                                out_file_handler.write(shell_command)

                            current_q_position = q_gap_end
                            current_t_position = t_gap_end

                        try:
                            line = next(chain_list)
                            line_number += 1
                        except StopIteration:
                            break
                else:
                    logging.info("invalid chain\n")

                    line = next(chain_list)
                    line_number += 1

                    while re.match(r"^\d+", line) is not None:
                        try:
                            line = next(chain_list)
                            line_number += 1
                        except StopIteration:
                            break

    logging.info("Done with reading gaps")
    logging.info(f"Gaps patched in this chain = {gap_count}")
    logging.info("\n")
    logging.info("\n")

    out_file_handler.close()


def make_shell_jobs(
    args: ArgsNamespace, current_chain_string: str
) -> tempfile._TemporaryFileWrapper[bytes]:
    """Makes a temp file with a jobList.

    Creates a temporary file to store shell commands for processing chain gaps.
    The file is created in the specified working directory and contains
    executable commands to run lastz alignments for each gap.

    Args:
        args: Parsed command-line arguments containing the working directory path
              and other configuration for shell job generation.
        current_chain_string: The chain alignment data as a string to be processed.

    Returns:
        A temporary file object (not deleted) containing the generated shell commands.
    """
    if not os.path.isdir(args.workdir):
        logging.error(f"ERROR! Working directory {args.workdir} does not exist.")
        sys.exit(1)

    # create temp file
    temp = tempfile.NamedTemporaryFile(
        prefix="tempCGFjobList", dir=args.workdir, delete=False
    )
    temp.close()

    # Find gaps and write corresponding jobs to a shell script
    make_shell_list(current_chain_string, temp.name, args)
    return temp


def run_all_shell(shell_file: ShellScriptPath) -> str:
    """Takes temp file with all shell commands to run and returns lastz output in a single string.

    Executes the generated shell script containing lastz alignment commands
    and returns the combined output from all commands. The output contains
    mini-chain alignments separated by LINE markers.

    Args:
        shell_file: Path to the shell script file containing executable commands.

    Returns:
        The decoded stdout output from running all shell commands as a string.
    """
    all_shell_command: str = f"bash {shell_file}"
    try:
        all_mini_chains: bytes = subprocess.check_output(all_shell_command, shell=True)
    except subprocess.CalledProcessError as shell_run:
        logging.error("shell command failed", shell_run.returncode, shell_run.output)
        sys.exit(1)

    all_mini_chains_decoded: str = all_mini_chains.decode()
    return all_mini_chains_decoded


def get_chain_block_from_lastz_output(
    all_mini_chains_split: list[str],
    cur_position: int,
) -> MiniChainBlockType:
    """Extracts a single block from lastz output starting at cur_position.

    Parses the output from running lastz alignments, which contains blocks
    of alignment data separated by LINE markers (e.g., "LINE123"). This function
    extracts one complete block delimited by LINE markers.

    The returned block contains:
    - LINE marker at start
    - Coordinate lines: block_len, TblockEnd, t_gap_end, real_q_block_end, real_q_gap_end
    - Chain alignment data
    - LINE marker at end

    Args:
        all_mini_chains_split: List containing line-wise lastz output including LINE###
                           statements that serve as block separators.
        cur_position: Starting position in the list where the block begins.
                      This position should contain a LINE### marker.

    Returns:
        List of strings representing one complete block starting with LINE###
        and ending with LINE# marker.

    Raises:
        ValueError: If the block is not properly separated by LINE markers
                   at either the start or end position.
    """

    position: int = cur_position
    start: int = position
    line: str = all_mini_chains_split[position]
    re_line = re.compile(r"LINE\d+")

    if re_line.match(line) is not None:
        position += 1
        line = all_mini_chains_split[position]

        while re_line.match(line) is None:
            position += 1
            line = all_mini_chains_split[position]

        if re_line.match(line) is not None:
            end: int = position
        else:
            raise ValueError(
                f"ERROR! all_mini_chains_split end separator line at "
                f"position {position} does not start with LINE..."
            )

    else:
        raise ValueError(
            f"ERROR! all_mini_chains_split start separator line at "
            f"position {str(position)} does not start with LINE..."
        )

    cur_block_list: MiniChainBlockType = all_mini_chains_split[start : (end + 1)]
    return cur_block_list


def take_first_chain_from_list(chain_list: list[str]) -> MiniChainOutputType:
    """Extracts the first chain from a list of chain lines.

    Parses a list of chain alignment lines and extracts the first complete
    chain record. A chain record consists of a header line starting with
    "chain" followed by alignment block lines containing numeric values.

    Args:
        chain_list: List of strings representing chain alignment data,
                   where each string is a line from the chain file.

    Returns:
        A tuple containing:
        - head_line: The chain header line (e.g., "chain 52633 chr...") or None if no chain found.
        - chain_content: List of alignment block lines for this chain, or None if no chain found.
    """
    head_line: ChainLineType | None = None
    chain_content: ChainContentType | None = None
    chain_start: int | None = None
    chain_end: int | None = None

    for pos in range(0, len(chain_list)):
        line: str = chain_list[pos]

        m = re.match(r"chain", line)
        if m is not None:
            head_line = line.strip("\n")

            pos += 1
            line = chain_list[pos]
            chain_start = pos

            while re.match(r"^\d+", line) is not None:
                pos += 1
                line = chain_list[pos]
            chain_end = pos

            break

    if chain_start is not None:
        chain_content = chain_list[chain_start:chain_end]
    return head_line, chain_content


def write_mini_chains_file(
    s: str,
    outfile: ShellScriptPath,
    enum: int,
) -> int:
    """Enumerates all mini chains and writes them to a file with enumeration tags.

    Processes mini chain data by adding enumeration identifiers to chain headers
    and appends the formatted chains to the output file. Each chain header is
    appended with a tab and enumeration number.

    Args:
        s: String containing mini chain alignment data, with chains separated
           by "chain" header lines.
        outfile: Path to the output file where enumerated chains will be written.
        enum: Starting enumeration number to assign to chain headers.

    Returns:
        The next enumeration number after processing all chains.
    """
    lines_list: list[str] = [f"{line}\n" for line in s.split("\n") if line]

    with open(outfile, "a") as ouf:
        for element in lines_list:
            if element.startswith("chain"):
                header_no_enum: str = " ".join(element.split()[:-1])
                element = f"{header_no_enum}\t{enum}\n"
                enum += 1
            ouf.write(element)

    return enum


def insert_chain_content(
    chain_content: ChainContentType,
    best_chain: str,
    block_len_a: str,
    t_block_end: str,
    t_gap_end: str,
    lo_q_block_end: str,
    lo_q_gap_end: str,
) -> list[str]:
    """Calculates new coordinates for a chain to be inserted and returns modified alignment blocks.

    After patching a chain gap with a new lastz alignment, this function recalculates
    the coordinates for the new chain content to ensure proper integration into the
    original chain structure. It handles both plus and minus strand alignments.

    The chain header format is:
    chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd chainId

    Args:
        chain_content: List of alignment block lines from the patched alignment
                      (excluding header and terminating lines).
        best_chain: The chain header line from the best lastz alignment result.
        block_len_a: Length of the first alignment block as a string.
        t_block_end: Target coordinate where the aligned block ends (string).
        t_gap_end: Target coordinate where the gap ends (string).
        lo_q_block_end: Query coordinate for the aligned block end (string).
        lo_q_gap_end: Query coordinate for the gap end (string).

    Returns:
        List of formatted alignment block lines with recalculated coordinates,
        ready to be inserted into the original chain.
    """
    t_lastz_start: int = int(best_chain.split()[5]) + 1
    t_lastz_end: int = int(best_chain.split()[6])

    q_lastz_start: int
    q_lastz_end: int
    if best_chain.split()[9] == "+":
        q_lastz_start = int(best_chain.split()[10]) + 1
        q_lastz_end = int(best_chain.split()[11])
    else:
        q_lastz_start = int(best_chain.split()[8]) - int(best_chain.split()[10])
        q_lastz_end = int(best_chain.split()[8]) - int(best_chain.split()[11]) + 1

        temp_q: str = lo_q_gap_end
        lo_q_gap_end = lo_q_block_end
        lo_q_block_end = temp_q

    blocks_to_add: list[str] = []

    first_q_gap: int
    last_q_gap: int
    if best_chain.split()[9] == "+":
        first_q_gap = abs(q_lastz_start - int(lo_q_block_end))
        last_q_gap = abs(int(lo_q_gap_end) - q_lastz_end)
    else:
        first_q_gap = abs(q_lastz_start - int(lo_q_block_end))
        last_q_gap = abs(int(lo_q_gap_end) - q_lastz_end)

    first_line: str = f"{str(block_len_a)}\t{str(t_lastz_start - int(t_block_end))}\t{str(first_q_gap)}\t"

    blocks_to_add.append(first_line)
    for i in range(0, len(chain_content) - 1):
        blocks_to_add.append(chain_content[i])

    chain_content_prelast: str = chain_content[len(chain_content) - 1].strip()
    last_line: str = f"{chain_content_prelast}\t{str(int(t_gap_end) - t_lastz_end)}\t{str(last_q_gap)}\t"
    blocks_to_add.append(last_line)
    return blocks_to_add


def fill_gaps_from_mini_chains(
    current_chain_lines: list[str],
    cur_mini_block_lines: list[str],
    args: ArgsNamespace,
    number_mini_chains: int,
    all_mini_chain_lines: list[str],
    start_time: float,
) -> None:
    """Processes initial chain and fills gaps with mini chains; writes to output file if provided.

    Iterates through the original chain alignment lines and inserts patched
    alignments at corresponding gap positions. Each mini-chain block contains
    coordinate information and the best alignment found by lastz. Only chains
    meeting the score threshold are inserted.

    The function processes LINE markers to identify where gap patches should
    be inserted, extracts the best alignment from lastz output, and recalculates
    coordinates for proper chain integration.

    Args:
        current_chain_lines: List of lines from the original chain file.
        cur_mini_block_lines: The first mini-chain block from lastz output
                            (with LINE markers at start and end).
        args: Parsed command-line arguments containing score thresholds
              and output file configuration.
        number_mini_chains: Total number of mini-chain blocks in the output.
        all_mini_chain_lines: All lines from lastz output split by newline.
        start_time: Timestamp when processing started (for logging elapsed time).
    """
    if args.output:
        ouf = open(args.output, "w")
    else:
        ouf = sys.stdout

    re_line_number = re.compile(r"LINE(\d+)")
    m = re_line_number.match(cur_mini_block_lines[0])
    if m is not None:
        next_line_number: int = int(m.group(1))
    else:
        raise ValueError(
            "ERROR! Could not extract line number from separator current miniChain block"
        )

    next_pos: int = 0
    for line_num in range(0, len(current_chain_lines)):
        line: str = current_chain_lines[line_num]

        if line_num == next_line_number:
            values_list: list[str] = cur_mini_block_lines[
                1 : (len(cur_mini_block_lines) - 1)
            ]
            coords: list[str] = values_list[:5]
            coords = [s.strip() for s in coords]

            next_pos = next_pos + len(cur_mini_block_lines) + 1
            if next_pos < number_mini_chains - 1:
                cur_mini_block_lines = get_chain_block_from_lastz_output(
                    all_mini_chain_lines, next_pos
                )
                m = re_line_number.match(cur_mini_block_lines[0])
                if m is not None:
                    next_line_number = int(m.group(1))
                else:
                    raise ValueError(
                        "ERROR! Could not extract line number from separator current miniChain block"
                    )

            best_chain: str | None
            chain_content: list[str] | None
            best_chain, chain_content = take_first_chain_from_list(values_list[5:])

            output_chain: str
            if best_chain is not None:
                if int(best_chain.split()[1]) >= args.scoreThreshold:
                    logging.info(f"Best lastz output chain = {best_chain}")

                    insert_block: list[str] = insert_chain_content(
                        chain_content, best_chain, *coords
                    )
                    output_chain = "\n".join(insert_block)
                    time_mark: float = time.time() - start_time
                    logging.info(f"--- {time_mark} seconds ---")
                else:
                    logging.info("lastz output chains have low score\n")
                    output_chain = line
            else:
                logging.info("lastz changed nothing in this block\n")
                output_chain = line
        else:
            output_chain = line

        if args.output:
            ouf.write(output_chain)
        else:
            print(output_chain)

    if args.output:
        ouf.close()


def main() -> None:
    """Main entry point for the repeat_filler script.

    Orchestrates the complete workflow of gap-filling chain alignments:
    1. Parses command-line arguments
    2. Reads the input chain file
    3. Generates shell commands to run lastz alignments for each gap
    4. Executes the shell commands to get mini-chain alignments
    5. Inserts the best alignments back into the original chain
    6. Outputs the modified chain to file or stdout
    """
    start_time: float = time.time()

    args: ArgsNamespace = parse_args()

    current_chain_string: str = get_chain_string(args)

    temp: tempfile._TemporaryFileWrapper[bytes] = make_shell_jobs(
        args, current_chain_string
    )

    all_mini_chains: str = run_all_shell(temp.name)

    os.unlink(temp.name)

    if all_mini_chains == "":
        if args.output:
            with open(args.output, "w") as file:
                file.write(current_chain_string)
        else:
            sys.stdout.write(current_chain_string)

        logging.info("Found no new blocks to insert in this chain. Done!")

    else:
        logging.info(
            "Found new blocks to insert in this chain. Filling gaps now . . . ."
        )

        next_pos: int = 0

        current_chain_lines: list[str] = [
            f"{i}\n" for i in current_chain_string.split("\n")
        ]

        all_mini_chain_lines: list[str] = [
            f"{i}\n" for i in all_mini_chains.split("\n")
        ]
        number_mini_chains: int = len(all_mini_chain_lines)

        cur_mini_block_lines: list[str] = get_chain_block_from_lastz_output(
            all_mini_chain_lines, next_pos
        )

        fill_gaps_from_mini_chains(
            current_chain_lines,
            cur_mini_block_lines,
            args,
            number_mini_chains,
            all_mini_chain_lines,
            start_time,
        )

    tot_time: float = time.time() - start_time
    logging.info(f"--- Final runtime: {tot_time} seconds ---")


if __name__ == "__main__":
    main()
