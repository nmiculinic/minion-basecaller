import logging
import os
import tempfile
from collections import Counter
import numpy as np
import pandas as pd
import operator
import pysam
from mincall import bioinf_utils as butil
from mincall.align_utils import split_aligments_in_sam
import shutil
from tqdm import tqdm
"""
 Script copied from https://github.com/isovic/samscripts/blob/master/src/consensus.py
 with minor modifications.
"""


def process_mpileup(
    name, alignments_path, reference_path, mpileup_path, coverage_threshold,
    output_prefix
):
    def _nlines(path):
        with open(path, 'r') as f:
            n_lines = sum(1 for _ in f)
        return n_lines

    n_lines = _nlines(mpileup_path)
    with open(mpileup_path, 'r') as fp:

        # snp_count,
        # insertion_count,
        # deletion_count,
        #
        # num_undercovered_bases,
        # num_called_bases,
        # num_correct_bases,
        # coverage_sum
        counts = np.zeros((7,))

        fp_variant = None
        fp_vcf = None

        if output_prefix:
            os.makedirs(output_prefix, exist_ok=True)

            variant_file = os.path.join(
                output_prefix, 'cov_%d.variant.csv' % coverage_threshold
            )
            fp_variant = open(variant_file, 'w')

            vcf_file = os.path.join(
                output_prefix, 'cov_%d.variant.vcf' % coverage_threshold
            )
            fp_vcf = open(vcf_file, 'w')

            fp_vcf.write('##fileformat=VCFv4.0\n')
            fp_vcf.write('##fileDate=20150409\n')
            fp_vcf.write('##source=none\n')
            fp_vcf.write('##reference=%s\n' % reference_path)
            fp_vcf.write(
                '##INFO=<ID=DP,Number=1,Type=Integer,Description="Raw Depth">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=TYPE,Number=A,Type=String,Description="Type of each allele (snp, ins, del, mnp, complex)">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=SB,Number=1,Type=Integer,Description="Phred-scaled strand bias at this position">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=DP4,Number=4,Type=Integer,Description="Counts for ref-forward bases, ref-reverse, alt-forward and alt-reverse bases">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=INDEL,Number=0,Type=Flag,Description="Indicates that the variant is an INDEL.">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=CONSVAR,Number=0,Type=Flag,Description="Indicates that the variant is a consensus variant (as opposed to a low frequency variant).">\n'
            )
            fp_vcf.write(
                '##INFO=<ID=HRUN,Number=1,Type=Integer,Description="Homopolymer length to the right of report indel position">\n'
            )
            fp_vcf.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
            fp_vcf.flush()

        i = 0
        j = 0
        num_bases_to_skip = 0

        for line in tqdm(fp, total=n_lines):
            if num_bases_to_skip > 0:
                num_bases_to_skip -= 1
                continue

            num_bases_to_skip, new_counts = process_mpileup_line(
                line, coverage_threshold, fp_variant, fp_vcf
            )
            counts += new_counts

            i += num_bases_to_skip
            i += 1
            j += 1

        fp.close()
        if fp_variant:
            fp_variant.close()

        if fp_vcf:
            fp_vcf.close()

        # transfrorm coverage sum to average coverage
        counts[-1] /= (i + 1)

        fields = [
            'alignments_file', 'mpileup_file', 'coverage_threshold',
            'snp_count', 'insertion_count', 'deletion_count',
            'num_undercovered_bases', 'num_called_bases', 'num_correct_bases',
            'average_coverage'
        ]
        values = [alignments_path, mpileup_path, coverage_threshold
                 ] + counts.tolist()
        report = pd.DataFrame([values], columns=fields, index=[name])
        report.num_called_bases = report.num_correct_bases + report.snp_count + report.insertion_count

        reference_len = len(butil.read_fasta(reference_path))
        for col in filter(lambda c: c.endswith('_count'), report.columns):
            new_col = col.replace('count', 'rate')
            report[new_col] = 100 * report[col] / report.num_called_bases

        report['correct_rate'
              ] = 100 * report.num_correct_bases / report.num_called_bases
        report['identity_percentage'
              ] = 100 * report.num_correct_bases / reference_len

        if output_prefix:
            summary_file = os.path.join(
                output_prefix, 'cov_%d.sum.vcf' % coverage_threshold
            )
            report.to_csv(summary_file, sep=';', index=False)
        return report


def process_mpileup_line(line, coverage_threshold, fp_variant, fp_vcf):
    def _fp_write(fp, line):
        if fp:
            fp.write(line + '\n')

    variant_write = lambda x: _fp_write(fp_variant, x)
    vcf_write = lambda x: _fp_write(fp_vcf, x)

    split_line = line.strip().split('\t')
    if len(split_line) < 5 or len(split_line) > 6:
        logging.error("Invalid mpileup line", line)
        return 0, np.zeros((7,))

    ref_name, position, ref_base, coverage, original_bases, *_ = split_line
    coverage = int(coverage)
    ref_base = ref_base.upper()

    bases = original_bases.replace('.', ref_base).replace(',', ref_base)

    base_counts = Counter()
    insertion_event_counts = Counter()
    deletion_event_counts = Counter()

    actual_insertion_count = 0
    actual_deletion_count = 0

    insertion_count = 0
    current_base_deletion_count = 0
    deletion_count = 0
    end_counts = 0
    snp_count = 0
    num_undercovered_bases = 0
    num_called_bases = 0
    num_correct_bases = 0
    coverage_sum = 0

    i = 0
    while i < len(bases):
        base = bases[i]

        if base == '^':
            i += 1

        elif base == '$':
            end_counts += 1

        elif base == '*':
            current_base_deletion_count += 1

        elif base == '-':
            j = i + 1
            while bases[j] in '0123456789':
                j += 1
            num_bases = int(bases[(i + 1):j])
            skip_bases = (j - i) + num_bases - 1
            deletion_count += 1
            deletion = bases[j:(j + num_bases)].upper()
            deletion_event_counts[deletion] += 1
            i += skip_bases

        elif base == '+':
            j = i + 1
            while bases[j] in '0123456789':
                j += 1
            num_bases = int(bases[(i + 1):j])
            skip_bases = (j - i) + num_bases - 1

            insertion_count += 1
            insertion = bases[j:(j + num_bases)].upper()
            insertion_event_counts[insertion] += 1
            i += skip_bases

        else:
            base_counts[bases[i].upper()] += 1
        i += 1

    non_indel_coverage_current_base = coverage - current_base_deletion_count

    if coverage < coverage_threshold:
        num_undercovered_bases += 1
        coverage_sum += coverage
        sorted_base_counts = sorted(
            base_counts.items(), key=operator.itemgetter(1)
        )

        variant_line = 'undercovered1\tpos = %s\tref = %s\tcoverage = %d\tbase_counts = %s\tinsertion_counts = %s\tdeletion_counts = %s' % (
            position, ref_name, int(coverage), str(sorted_base_counts),
            str(insertion_event_counts), str(deletion_event_counts)
        )
        variant_write(variant_line)

        ### VCF output ###
        qual = 1000
        info = 'DP=%d;TYPE=snp' % coverage
        ref_field = ref_base
        alt_field = 'N'
        vcf_line = '%s\t%s\t.\t%s\t%s\t%d\tPASS\t%s' % (
            ref_name, position, ref_field, alt_field, qual, info
        )
        vcf_write(vcf_line)

    else:
        num_called_bases += 1
        coverage_sum += coverage
        most_common_base_count = 0

        sorted_base_counts = sorted(
            base_counts.items(), key=operator.itemgetter(1)
        )
        try:
            most_common_base_count = sorted_base_counts[-1][1]
        except:
            pass

        is_good = False
        for base_count in sorted_base_counts:
            if base_count[1] == most_common_base_count:
                if base_count[0] == ref_base:
                    is_good = True
                    break
        if not is_good:
            if len(sorted_base_counts) > 0:
                snp_count += 1
                variant_line = 'SNP\tpos = %s\tref = %s\tcoverage = %d\tnon_indel_cov_curr = %d\tmost_common_base_count = %d\tref_base = %s\tcons_base = %s\tbase_counts = %s\tinsertion_counts = %s\tdeletion_counts = %s\t%s' % (
                    position, ref_name, int(coverage),
                    non_indel_coverage_current_base, most_common_base_count,
                    ref_base, ('{}') if (len(sorted_base_counts) == 0) else
                    (str(sorted_base_counts[-1][0])), str(sorted_base_counts),
                    str(insertion_event_counts), str(deletion_event_counts),
                    line.strip()
                )
                variant_write(variant_line)

                ### VCF output ###
                alt_base = ('{}') if (len(sorted_base_counts) == 0
                                     ) else (str(sorted_base_counts[-1][0]))
                qual = 1000
                info = 'DP=%d;TYPE=snp' % (coverage)
                ref_field = ref_base
                alt_field = alt_base
                vcf_line = '%s\t%s\t.\t%s\t%s\t%d\tPASS\t%s' % (
                    ref_name, position, ref_field, alt_field, qual, info
                )
                vcf_write(vcf_line)
                ##################

        else:
            num_correct_bases += 1

    non_indel_coverage_next_base = coverage - end_counts - deletion_count - insertion_count
    skip = 0
    if (
        non_indel_coverage_next_base + deletion_count + insertion_count
    ) > coverage_threshold:
        if len(insertion_event_counts.keys()) > 0:
            sorted_insertion_counts = sorted(
                insertion_event_counts.items(), key=operator.itemgetter(1)
            )
            most_common_insertion_count = sorted_insertion_counts[-1][1]
            most_common_insertion_length = len(sorted_insertion_counts[-1][0])
            insertion_unique = True if (
                sum([
                    int(insertion_count[1] == most_common_insertion_count)
                    for insertion_count in sorted_insertion_counts
                ]) == 1
            ) else False
        else:
            most_common_insertion_count = 0
            most_common_insertion_length = 0
            insertion_unique = False

        if len(deletion_event_counts.keys()) > 0:
            sorted_deletion_counts = sorted(
                deletion_event_counts.items(), key=operator.itemgetter(1)
            )
            most_common_deletion_count = sorted_deletion_counts[-1][1]
            most_common_deletion_length = len(sorted_deletion_counts[-1][0])
            deletion_unique = True if (
                sum([
                    int(deletion_count[1] == most_common_deletion_count)
                    for deletion_count in sorted_deletion_counts
                ]) == 1
            ) else False
        else:
            most_common_deletion_count = 0
            most_common_deletion_length = 0
            deletion_unique = False

        if most_common_insertion_count > most_common_deletion_count and most_common_insertion_count > non_indel_coverage_next_base:
            if insertion_unique:
                actual_insertion_count += 1
                num_called_bases += most_common_insertion_length

                try:
                    temp_sorted_bc = sorted_base_counts[-1][0]
                except:
                    temp_sorted_bc = 0

                variant_line = 'ins\tpos = %s\tref = %s\tnon_indel_cov_next = %d\tnon_indel_cov_curr = %d\tmost_common_insertion_count = %d\tref_base = %s\tcons_base = %s\tbase_counts = %s\tinsertion_counts = %s\tdeletion_counts = %s\t%s' % (
                    position, ref_name, non_indel_coverage_next_base,
                    non_indel_coverage_current_base,
                    most_common_insertion_count, ref_base, temp_sorted_bc,
                    str(sorted_base_counts), str(insertion_event_counts),
                    str(deletion_event_counts), line.strip()
                )
                variant_write(variant_line)

                ### VCF output ###
                qual = 1000
                info = 'DP=%d;TYPE=ins' % coverage
                ref_field = ref_base
                alt_field = '%s%s' % (ref_base, sorted_insertion_counts[-1][0])
                vcf_line = '%s\t%s\t.\t%s\t%s\t%d\tPASS\t%s' % (
                    ref_name, position, ref_field, alt_field, qual, info
                )
                vcf_write(vcf_line)
                ##################

        elif most_common_deletion_count > most_common_insertion_count and most_common_deletion_count > non_indel_coverage_next_base:
            if deletion_unique:
                actual_deletion_count += 1
                variant_line = 'del\tpos = %s\tref = %s\tnon_indel_cov_next = %d\tnon_indel_cov_curr = %d\tmost_common_deletion_count = %d\tref_base = %s\tcons_base = %s\tbase_counts = %s\tinsertion_counts = %s\tdeletion_counts = %s\t%s' % (
                    position, ref_name, non_indel_coverage_next_base,
                    non_indel_coverage_current_base, most_common_deletion_count,
                    ref_base, sorted_base_counts[-1][0],
                    str(sorted_base_counts), str(insertion_event_counts),
                    str(deletion_event_counts), line.strip()
                )
                variant_write(variant_line)

                ### VCF output ###
                qual = 1000
                info = 'DP=%d;TYPE=del' % coverage
                ref_field = '%s%s' % (ref_base, sorted_deletion_counts[-1][0])
                alt_field = ref_base
                vcf_line = '%s\t%s\t.\t%s\t%s\t%d\tPASS\t%s' % (
                    ref_name, position, ref_field, alt_field, qual, info
                )
                vcf_write(vcf_line)
                ##################
                skip = most_common_deletion_length

    cnts = [
        snp_count, actual_insertion_count, actual_deletion_count,
        num_undercovered_bases, num_called_bases, num_correct_bases,
        coverage_sum
    ]

    return skip, np.array(cnts)


def get_consensus_report(
    name,
    sam_path,
    ref_path,
    is_circular,
    coverage_threshold=0,
    report_out_dir=None,
    tmp_files_dir=None
):
    basename = os.path.basename(sam_path)
    file_name, ext = os.path.splitext(basename)

    out_dir = tmp_files_dir
    keep_tmp_files = tmp_files_dir is not None
    if not keep_tmp_files:
        out_dir = tempfile.mkdtemp()

    os.makedirs(out_dir, exist_ok=True)
    tmp_sam_path = os.path.join(out_dir, file_name + '_tmp.sam')
    tmp_bam_path = os.path.join(out_dir, file_name + '_tmp.bam')
    bam_path = os.path.join(out_dir, file_name + '.bam')
    mpileup_path = bam_path + '.bam.mpilup'

    logging.info("Split long aligments")
    split_aligments_in_sam(sam_path, tmp_sam_path)

    logging.info("Converting sam to bam")
    pysam.view('-S', tmp_sam_path, '-b', '-o', tmp_bam_path, catch_stdout=False)

    logging.info("Sorting bam file")
    pysam.sort(tmp_bam_path, '-o', bam_path, catch_stdout=False)

    logging.info("Creating bam index")
    pysam.index(bam_path, '-b')

    logging.info("Creating mpileup")

    mpileup_flags = ['-A', '-B', '-Q', '0']
    if is_circular:
        # use secondary aligments as well
        mpileup_flags.extend(['--ff', '0'])

    pysam.mpileup(
        *mpileup_flags,
        '-f',
        ref_path,
        bam_path,
        '-o',
        mpileup_path,
        catch_stdout=False
    )

    logging.info("Generating consensus and report")
    report = process_mpileup(
        name, sam_path, ref_path, mpileup_path, coverage_threshold,
        report_out_dir
    )

    if not keep_tmp_files:
        logging.info("Cleaning tmp files")
        shutil.rmtree(out_dir)

    return report
